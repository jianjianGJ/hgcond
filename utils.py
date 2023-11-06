import os
import torch
import random
import numpy as np
import math
from models import HeteroSGC,HeteroGCN,HeteroSAGE,HAN,HeteroGCN2
import torch.nn.functional as F
from copy import deepcopy
from torch_sparse import sum as sparsesum, mul, SparseTensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score  
#%%
def print_dict(data_dict, start=0):
    prefix = start*'-'
    for k,v in data_dict.items():
        if type(v)==dict:
            print(f'{prefix}{k}')
            print_dict(v, start+len(str(k)))
        else:
            if type(v)==list:
                print(f'{prefix}{k}:[{v[0]},...]')
            elif type(v)==np.ndarray:
                print(f'{prefix}{k}:array{v.shape}')
            elif type(v)==str and len(v)>10:
                print(f'{prefix}{k}:some text')
            else:
                print(f'{prefix}{k}:{v}')
def get_distribution(values, name=None, unit=0.01):
    values_min, values_max = values.min().item(), values.max().item()
    low, up = values_min, values_min+unit
    x_list, count_list = [], []
    while low < values_max:
        mid = (low + up) / 2
        x_list.append(mid)
        count = torch.bitwise_and(low < values, values <= up)
        count_list.append(count.sum().item())
        low = up
        up = low + unit
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_title(name)
    ax.plot(x_list, count_list)
    return fig
def get_count(values, upper):
    count = F.one_hot(values, num_classes=upper).sum(0)
    return count
def plotlog(log):
    fig, ax = plt.subplots()
    ax.plot(log)
    fig.savefig('losses.png')
def seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def exp_exists(path, version):
    exp_list_ = os.listdir(path)
    exp_list = [exp[:-7] for exp in exp_list_]
    if version in exp_list:
        return True
    else:
        return False
# def asymmetric_gcn_norm(adj_t):
#     if isinstance(adj_t, SparseTensor):
#         if not adj_t.has_value():
#             adj_t = adj_t.fill_value(1.)
#         deg_dst = sparsesum(adj_t, dim=1)
#         mask = [deg_dst>0.] 
#         deg_dst_real = deg_dst[mask]
#         deg_dst[mask] = 1./deg_dst_real
        
#         adj_t = mul(adj_t, deg_dst.view(-1, 1))
#     else:
#         deg_dst = adj_t.sum(1)
#         mask = [deg_dst>0.] 
#         deg_dst_real = deg_dst[mask]
#         deg_dst[mask] = 1./deg_dst_real
#         adj_t = adj_t*deg_dst.view(-1, 1)
#     return adj_t
def asymmetric_gcn_norm(adj_t):
    if isinstance(adj_t, SparseTensor):
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.)
        deg_src = sparsesum(adj_t, dim=0)+0.00001
        deg_src_inv_sqrt = deg_src.pow_(-0.5)
        deg_src_inv_sqrt.masked_fill_(deg_src_inv_sqrt == float('inf'), 0.)
        deg_dst = sparsesum(adj_t, dim=1)+0.00001
        deg_dst_inv_sqrt = deg_dst.pow_(-0.5)
        deg_dst_inv_sqrt.masked_fill_(deg_dst_inv_sqrt == float('inf'), 0.)
        
        adj_t = mul(adj_t, deg_dst_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_src_inv_sqrt.view(1, -1))
    else:
        deg_src = adj_t.sum(0)+0.00001
        deg_src_inv_sqrt = deg_src.pow_(-0.5)
        deg_src_inv_sqrt.masked_fill_(deg_src_inv_sqrt == float('inf'), 0.)
        deg_dst = adj_t.sum(1)+0.00001
        deg_dst_inv_sqrt = deg_dst.pow_(-0.5)
        deg_dst_inv_sqrt.masked_fill_(deg_dst_inv_sqrt == float('inf'), 0.)
        adj_t = adj_t*deg_dst_inv_sqrt.view(-1, 1)
        adj_t = adj_t*deg_src_inv_sqrt.view(1, -1)
    return adj_t
def adj_t_dict_to_edge_index_dict(adj_t_dict):
    edge_index_dict = {}
    for edge_type,adj_t in adj_t_dict.items():
        if isinstance(adj_t, SparseTensor):
            edge_index_dict[edge_type] = torch.vstack([adj_t.coo()[1],adj_t.coo()[0]])
        else:
            edge_index_inv = adj_t.to_sparse_coo().indices()
            edge_index_dict[edge_type] = torch.vstack([edge_index_inv[1],edge_index_inv[0]])
    return edge_index_dict
def get_GNN(modelname):
    if modelname=='HeteroSGC':
        GNN = HeteroSGC
    elif modelname=='HeteroGCN':
        GNN = HeteroGCN
    elif modelname=='HeteroSAGE':
        GNN = HeteroSAGE
    elif modelname=='HAN':
        GNN = HAN
    elif modelname=='HeteroGCN2':
        GNN = HeteroGCN2
        

    return GNN
def custom_loss_function(x, labels):
    epsilon = 1 - math.log(2)
    y = F.cross_entropy(x, labels, reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)
def evalue_model(model, x_dict, adj_t_dict, y, test_mask):
    model.eval()
    logits = model(x_dict, adj_t_dict)[test_mask]
    labels = y[test_mask].cpu()
    preds = logits.argmax(1).cpu()
    acc = (labels == preds).sum()/test_mask.sum()
    acc = acc.item()
    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    return acc,f1_micro,f1_macro
def train_model(model, opt_parameter,optimizer, x_syn_dict, adj_t_syn_dict, y_syn, mask_syn):
    for epoch in range(1, opt_parameter['epochs_basic_model']+1):
        model.train()
        optimizer.zero_grad()
        out = model(x_syn_dict, adj_t_syn_dict)[mask_syn]
        loss = F.nll_loss(out, y_syn[mask_syn]) # nll_loss  cross_entropy
        loss.backward()
        optimizer.step()
def train_model_ealystop(model, opt_parameter, x_dict, adj_t_dict, y,
                         train_mask, val_mask):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_parameter['lr'], weight_decay=opt_parameter['weight_decay'])

    best_val_acc = 0
    for epoch in tqdm(range(1, opt_parameter['epochs']+1),desc='Training',ncols=80):
        model.train()
        optimizer.zero_grad()
        out = model(x_dict, adj_t_dict)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        
        model.eval()
        pred = model(x_dict, adj_t_dict).argmax(dim=-1)
        val_acc = (pred[val_mask] == y[val_mask]).sum() / val_mask.sum()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            weights = deepcopy(model.state_dict())
    model.load_state_dict(weights)
    return best_val_acc
def train_model_ealystop_patience(model, opt_parameter, x_dict, adj_t_dict, y,
                         train_mask, val_mask):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_parameter['lr'], weight_decay=opt_parameter['weight_decay'])
    max_patience = 5
    patience = 0
    best_val_acc = 0
    for epoch in tqdm(range(1, opt_parameter['epochs']+1),desc='Training',ncols=80):
        model.train()
        optimizer.zero_grad()
        out = model(x_dict, adj_t_dict)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        
        model.eval()
        pred = model(x_dict, adj_t_dict).argmax(dim=-1)
        val_acc = (pred[val_mask] == y[val_mask]).sum() / val_mask.sum()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            weights = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience == max_patience:
            break
    model.load_state_dict(weights)
    return best_val_acc
def lazy_initialize(model, x_dict, adj_t_dict):
    with torch.no_grad(): 
        model(x_dict, adj_t_dict)
def related_parameters(basic_model, x_dict, adj_t_dict, y, train_mask):
    output = basic_model(x_dict, adj_t_dict)
    loss_real = F.nll_loss(output[train_mask], y[train_mask])
    gw_reals = torch.autograd.grad(loss_real, basic_model.parameters(), allow_unused=True)
    parameters = []
    for i,p in enumerate(basic_model.parameters()):
        if gw_reals[i]!=None and p.data.dim()==2:
            parameters.append(p)
    return parameters
def getsize_mb(elements):
    size = 0
    for e in elements:
        if type(e)==dict:
            for v in e.values():
                if type(v)==SparseTensor:
                    row, col, value = v.coo()
                    size += row.element_size()*row.nelement()
                    size += col.element_size()*col.nelement()
                    size += value.element_size()*value.nelement()
                else:
                    size += v.element_size()*v.nelement()
        else:
            size += e.element_size()*e.nelement()
    return size/1024/1024 