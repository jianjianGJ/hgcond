import torch
from models import HeteroSGC,HeteroGCN,HeteroSAGE,HAN,HeteroGCN2
import torch.nn.functional as F
from copy import deepcopy
from torch_sparse import sum as sparsesum, mul, SparseTensor
from tqdm import tqdm
from sklearn.metrics import f1_score  
#%%
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
