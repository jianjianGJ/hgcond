import os
from copy import deepcopy
from tqdm import tqdm
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import orthogonal_
from torch_scatter import scatter_mean, scatter_sum
from sklearn.cluster import BisectingKMeans 
import prettytable as pt
from utils import (get_GNN, train_model_ealystop, lazy_initialize, 
                   evalue_model, related_parameters, train_model,
                   asymmetric_gcn_norm)
#%% 
def match_loss(gw_syns, gw_reals):
    dis = 0
    for ig in range(len(gw_reals)):
        gw_real = gw_reals[ig]
        gw_syn = gw_syns[ig]
        if gw_syn.dim()==2:
            dis += distance_w(gw_real, gw_syn)
    return dis
def distance_w(gwr, gws):
    gwr = gwr.T
    gws = gws.T
    dis_weight = torch.mean(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis
#%% Tool fuctions
def gcond_initialize(data, num_syn_dict):
    node_types,edge_type = data.node_types,data.edge_types
    target_node_type = data.target_node_type
    x_dict, adj_t_dict, y = data.x_dict, data.adj_t_dict, data[target_node_type].y.clone()
    train_mask = data[target_node_type].train_mask
    
    x_initial_dict = {}
    for node_type in node_types:
        num_full = x_dict[node_type].shape[0]
        num_syn = num_syn_dict[node_type]
        if node_type!=target_node_type:
            random_selected = torch.randperm(num_full)[:num_syn]
            x_initial_dict[node_type] = x_dict[node_type][random_selected]
        else:
            labeled_index = torch.arange(num_full)[train_mask]
            num_labeld = labeled_index.shape[0]
            random_selected = labeled_index[torch.randint(num_labeld,(num_syn,))]
            x_initial_dict[node_type] = x_dict[node_type][random_selected]
            y_syn = y[random_selected]
            mask_syn = y_syn>-99
    
    indices_dict = {}# edge_index format adj
    for edge_type, adj_t in adj_t_dict.items():
        src_type, _, dst_type = edge_type
        dst_num, src_num = num_syn_dict[dst_type], num_syn_dict[src_type]
        edge_cout = torch.ones(dst_num, src_num)# full connected graph
        indices_dict[edge_type] = torch.LongTensor(edge_cout.to_sparse().indices())
    return x_initial_dict, indices_dict, y_syn, mask_syn

def cluster_initialize(data, num_syn_dict, model_name, architecture, opt_parameter, file_path):
    device = data[data.target_node_type].y.device
    target_node_type, num_classes = data.target_node_type, data.num_classes
    x_dict, adj_t_dict, y = data.x_dict, data.adj_t_dict, data[target_node_type].y
    train_mask = data[target_node_type].train_mask
    val_mask =  data[target_node_type].val_mask
    #-------------------------------get cluster information
    if os.path.exists(file_path):#load pre-calculated results
        cluster_dict, cluster_adi_dict, cluster_y_count = torch.load(file_path) #device is cpu
    else:
        print('Getting embedding for clustering:')
        model = get_GNN(model_name)(**architecture, 
                                    out_channels=num_classes, 
                                    node_types=data.node_types, 
                                    edge_types=data.edge_types,
                                    target_node_type=target_node_type).to(device)
        lazy_initialize(model, x_dict, adj_t_dict)
        time_start=time.time()
        acc = train_model_ealystop(model, opt_parameter, x_dict, adj_t_dict, y, train_mask, val_mask)
        time_end=time.time()
        time_used = time_end-time_start
        print(f'Embedding obtained: acc:{acc:.4f} time for 100 epochs:{time_used:.4f}')
        _, h_dict = model(x_dict, adj_t_dict, get_embeddings=True)
        h_dict = {node_type:h.detach().cpu() for node_type,h in h_dict.items()}
        cluster_dict = {}# cluster result (cluster labels) for all node_types
        print('Clustering for initialization.')
        for node_type, h in h_dict.items():
            k_means = BisectingKMeans(n_clusters=num_syn_dict[node_type], random_state=0)
            k_means.fit(h)
            cluster_dict[node_type] = torch.LongTensor(k_means.predict(h))
            if node_type==target_node_type:
                y_train = y.clone()
                y_train[~train_mask] = -1
                y_onehot = F.one_hot(y_train+1, num_classes=num_classes+1)[:,1:].cpu()
                #cluster_y_count saves the numbers of all type of labels in each cluster
                cluster_y_count = scatter_sum(y_onehot, cluster_dict[node_type], dim=0)
        cluster_adi_dict = {}
        # cluster_adi_dict[edge_type][i][j] is the number of links between cluster dst_type-i and src_type-j
        for edge_type, adj_t in adj_t_dict.items():
            src_type, _, dst_type = edge_type
            src_num, dst_num = cluster_dict[src_type].max()+1, cluster_dict[dst_type].max()+1
            cluster_adi_dict[edge_type] = torch.zeros(dst_num, src_num)
            row, col, v = adj_t.coo()
            c_src, c_dst = cluster_dict[src_type], cluster_dict[dst_type]
            c_dst_row = c_dst[row]
            c_src_col = c_src[col]
            for i in range(dst_num):
                mask_i = c_dst_row==i
                connected = c_src_col[mask_i]
                connected = connected[connected>-1]
                cluster_adi_dict[edge_type][i] = F.one_hot(connected,num_classes=src_num).sum(0)
        torch.save((cluster_dict, cluster_adi_dict, cluster_y_count), file_path)
    # -------------get initialization information from cluster information
    x_initial_dict = {}
    for node_type in data.node_types:
        clusters = cluster_dict[node_type]
        x_initial_dict[node_type] = scatter_mean(x_dict[node_type].cpu(), clusters, dim=0)
        if node_type==target_node_type:
            y_train = y.clone()
            y_train[~train_mask] = -1
            y_onehot = F.one_hot(y_train+1, num_classes=num_classes+1)[:,1:].cpu()
            count = scatter_sum(y_onehot, clusters, dim=0)
            y_syn = count.argmax(dim=1)
            mask_syn = count.sum(1)>0
            y_syn[~mask_syn] = -1 # no label syn-nodes
    indices_dict = {}
    for edge_type in data.edge_types:
        edge_cout = cluster_adi_dict[edge_type]
        # edge_cout[edge_cout<1] = 0
        indices_dict[edge_type] = torch.LongTensor(edge_cout.to_sparse().indices())
    print('Initialization obtained.')
    return x_initial_dict, indices_dict, y_syn, mask_syn
#%% GraphSynthesizer

class GraphSynthesizer(nn.Module):

    def __init__(self, data, cond_rate, feat_init='cluster', edge_hidden_channels=None):
        super(GraphSynthesizer, self).__init__()
        self.device = data[data.target_node_type].x.device
        self.name = data.name
        self.cond_rate = cond_rate
        self.target_node_type = data.target_node_type
        self.edge_hidden_channels = edge_hidden_channels
        self.node_types = data.node_types
        self.edge_types = data.edge_types
        self.num_classes = data.num_classes
        #-------------------------------------------------------------------------
        self.num_syn_dict = {}
        for node_type,x in data.x_dict.items():
            num_syn = max(int(x.shape[0]*cond_rate),1) 
            self.num_syn_dict[node_type] = num_syn
        #-------------------------------------------------------------------------
        if feat_init=='cluster':
            model_name = 'HeteroSGC'
            architecture = {'hidden_channels':64,
                            'num_layers':3}
            opt_parameter = {'epochs':100,
                             'lr':0.005,
                             'weight_decay':0.001}
            save_path = './clusters/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = save_path + f'/{data.name}-{cond_rate}.clusters'
            x_initial_dict, indices_dict, y_syn, mask_syn = cluster_initialize(data, self.num_syn_dict, model_name, architecture, opt_parameter, file_path)
        elif feat_init=='sample':
            x_initial_dict, indices_dict, y_syn, mask_syn = gcond_initialize(data, self.num_syn_dict)
        
        
        self.x_initial_dict = {k:v.to(self.device) for k,v in x_initial_dict.items()}
        self.indices_dict = {k:v.to(self.device) for k,v in indices_dict.items()}
        self.y_syn, self.mask_syn = y_syn.to(self.device), mask_syn.to(self.device)
        #-------------------------------------------------------------------------
        self.x_syn_dict = {}
        for node_type,x in data.x_dict.items():
            num_syn = self.num_syn_dict[node_type]
            self.x_syn_dict[node_type] = nn.Parameter(torch.FloatTensor(num_syn, x.shape[1]).to(self.device))#
            self.x_syn_dict[node_type].data.copy_(self.x_initial_dict[node_type])
        #-------------------------------------------------------------------------
        if self.edge_hidden_channels is not None:
            self.edge_mlp_dict = {}
            for edge_type in data.edge_index_dict.keys():
                src_type, _, dst_type = edge_type
                in_channels = data.x_dict[src_type].shape[1] + data.x_dict[dst_type].shape[1]
                self.edge_mlp_dict[edge_type] = \
                    nn.Sequential(nn.Linear(in_channels, edge_hidden_channels),
                                    nn.BatchNorm1d(edge_hidden_channels),
                                    nn.ReLU(),
                                    # nn.Linear(edge_hidden_channels, edge_hidden_channels),
                                    # nn.BatchNorm1d(edge_hidden_channels),
                                    nn.ReLU(),
                                    nn.Linear(edge_hidden_channels, 1)).to(self.device)
        self.get_adj_t_syn_dict()
    def x_parameters(self):
        return list(self.x_syn_dict.values())
    def adj_parameters(self):
        parameters = []
        if self.edge_hidden_channels is not None:
            for edge_mlp in self.edge_mlp_dict.values():
                parameters.extend(edge_mlp.parameters())
        return parameters
    def get_x_syn_dict(self):
        return self.x_syn_dict
    def get_adj_t_syn_dict(self):
        self.adj_t_syn_dict = {}
        for edge_type in self.edge_types:
            src_type, _, dst_type = edge_type
            num_src, num_dst = self.num_syn_dict[src_type], self.num_syn_dict[dst_type]
            if isinstance(num_src, torch.Tensor):
                num_src = int(num_src.sum().item())
            if isinstance(num_dst, torch.Tensor):
                num_dst = int(num_dst.sum().item())
            adj_t_syn = torch.zeros(size=(num_dst, num_src), device=self.device)
            indices = self.indices_dict[edge_type]
            if indices.shape[1] <= 1:
                continue
            row, col = indices[0], indices[1]
            if self.edge_hidden_channels is None:
                adj_t_syn[row, col] = 1.
            else:
                adj_t_syn[row,col] = torch.sigmoid(self.edge_mlp_dict[edge_type](
                                torch.cat([self.x_syn_dict[dst_type][row], self.x_syn_dict[src_type][col]], dim=1)).flatten())
            
            adj_t_syn = asymmetric_gcn_norm(adj_t_syn)
            self.adj_t_syn_dict[edge_type] = adj_t_syn
        return self.adj_t_syn_dict
    def get_x_syn_detached_dict(self):
        x_syn_dict = {node_type:x.detach() for node_type,x in self.x_syn_dict.items()}
        return x_syn_dict
    def get_adj_t_syn_detached_dict(self):
        adj_t_syn_dict = {edge_type:adj_t.detach() for edge_type,adj_t in self.adj_t_syn_dict.items()}
        return adj_t_syn_dict
    def __repr__(self):
        name = f'{self.name}-{self.cond_rate}'
        table = pt.PrettyTable(['Name', 'shape', 'mean', 'sparsity', 'labels'])
        for node_type,x in self.x_syn_dict.items():
            shape = str((x.shape[0],x.shape[1]))
            nnz = (x>0).sum().item()
            mean_ = f'{x.sum().item()/nnz:.2f}'
            sparsity = f'{nnz/x.shape[0]/x.shape[1]:.2f}'
            if node_type==self.target_node_type:
                label_numpy = F.one_hot(self.y_syn[self.mask_syn],num_classes=self.num_classes).sum(0).cpu().numpy()
                label = str(label_numpy)
                if len(label)>15:
                    label = f'[{label_numpy[0]},...,{label_numpy[-1]}]'
            else:
                label = ''
            table.add_row([node_type, shape, mean_, sparsity, label])
        table.add_row(['','','','',''])
        for edge_type, adj_t_syn in self.adj_t_syn_dict.items():
            nnz = (adj_t_syn>0).sum().item()
            mean_ = f'{adj_t_syn.sum().item()/nnz:.2f}'
            shape = str((adj_t_syn.shape[0],adj_t_syn.shape[1]))
            sparsity = f'{nnz/adj_t_syn.shape[0]/adj_t_syn.shape[1]:.2f}'
            label = ''
            table.add_row([edge_type, shape, mean_, sparsity, label])
        table.title = name
        info = table.get_string()        
        return info
#%%
def evalue_hgcond(num_evalue, data, x_syn_dict, adj_t_syn_dict, y_syn, mask_syn, 
                  model_name, model_architecture, model_train, loss_fn = F.cross_entropy):
    node_types = data.node_types                                # cross_entropy nll_loss
    edge_types = data.edge_types
    target_node_type, num_classes = data.target_node_type, data.num_classes
    x_dict, adj_t_dict, y = data.x_dict, data.adj_t_dict, data[target_node_type].y
    train_mask = data[target_node_type].train_mask
    val_mask =  data[target_node_type].val_mask
    test_mask = data[target_node_type].test_mask
    device = train_mask.device
    
    x_syn_dict = {k:v.to(device) for k,v in x_syn_dict.items()}
    adj_t_syn_dict = {k:v.to(device) for k,v in adj_t_syn_dict.items()}
    y_syn = y_syn.to(device)
    mask_syn = mask_syn.to(device)
    
    val_model = get_GNN(model_name)(**model_architecture, 
                                out_channels=num_classes, 
                                node_types=node_types, 
                                edge_types=edge_types,
                                target_node_type=target_node_type).to(device)
    lazy_initialize(val_model, x_dict, adj_t_dict)
    
    max_patience = 10
    trig_early_stop = True
    accs, f1_micros, f1_macros = [], [], []
    for i in range(num_evalue):
        best_acc = 0.
        for j in tqdm(range(model_train['epochs']),desc='Traning', ncols=80):
            if trig_early_stop:
                patience = 0
                trig_early_stop = False
                val_model.reset_parameters()
                optimizer_val_model = torch.optim.Adam(val_model.parameters(), lr=model_train['lr'])
            val_model.train()
            optimizer_val_model.zero_grad()
            logits_train = val_model(x_syn_dict, adj_t_syn_dict)[mask_syn]
            loss = loss_fn(logits_train, y_syn[mask_syn])# cross_entropy nll_loss
            loss.backward()
            optimizer_val_model.step()
            with torch.no_grad():
                val_model.eval()
                logits_val = val_model(x_dict, adj_t_dict)[val_mask]
                acc = (logits_val.argmax(1) == y[val_mask]).sum()/logits_val.shape[0]
                if acc > best_acc:
                    best_acc = acc
                    patience = 0
                    weights = deepcopy(val_model.state_dict())
                else:
                    patience += 1
                if patience == max_patience:
                    trig_early_stop = True
        #--------------------------------------------------------------------------
        val_model.load_state_dict(weights)
        acc,f1_micro,f1_macro = evalue_model(val_model, x_dict, adj_t_dict, y, test_mask)
        accs.append(acc)
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)
    return accs,f1_micros,f1_macros

#%%
class Orth_Initializer:
    def __init__(self, model):
        self.cach_list = []
        self.index_list = []
        for p in model.parameters():
            if p.dim()==2:
                n_row, n_col = p.shape[0], p.shape[1]
                base = torch.empty(n_row,n_row,n_col)
                for i in range(n_col):
                    orthogonal_(base[:,:,i])
                self.cach_list.append(base)
                self.index_list.append(0)
    def next_init(self):
        init_list = []
        for i,cach in enumerate(self.cach_list):
            if self.index_list[i] < cach.shape[0]:
                init_list.append(cach[self.index_list[i]])
                self.index_list[i] += 1
            else:
                for j in range(cach.shape[2]):
                    orthogonal_(self.cach_list[i][:,:,j])
                init_list.append(self.cach_list[i][0])
                self.index_list[i] = 1
        return init_list
def hgcond(data, cond_rate, feat_init, para_init, basicmodel, model_architecture, cond_train):
    #################################### get data info
    target_node_type, num_classes = data.target_node_type, data.num_classes
    node_types, edge_types = data.node_types, data.edge_types
    x_dict, adj_t_dict, y = data.x_dict, data.adj_t_dict, data[target_node_type].y
    train_mask = data[target_node_type].train_mask
    #################################### basic model
    basic_model = get_GNN(basicmodel)(**model_architecture, 
                                out_channels=num_classes, 
                                node_types=node_types, 
                                edge_types=edge_types, 
                                target_node_type=target_node_type).to(data[target_node_type].y.device)
    lazy_initialize(basic_model, x_dict, adj_t_dict)
    if para_init=='orth':
        orth_initi = Orth_Initializer(basic_model)
    #################################### GraphSynthesizer
    graphsyner = GraphSynthesizer(data, cond_rate, feat_init, edge_hidden_channels=64)
    y_syn = graphsyner.y_syn
    mask_syn = graphsyner.mask_syn
    optimizer_cond = torch.optim.Adam(graphsyner.x_parameters()+graphsyner.adj_parameters(), 
                                    lr=cond_train['lr'])
    parameters = related_parameters(basic_model, x_dict, adj_t_dict, y, train_mask)
    #################################### condensation stage
    losses_log = []
    smallest_loss = 99999.
    for initial_i in tqdm(range(cond_train['epochs_initial']), 
                          desc='Condasention', ncols=80):
        if para_init=='orth':
            basic_model.reset_parameters(orth_initi.next_init())
        else:
            basic_model.reset_parameters()
        optimizer_basic_model = torch.optim.Adam(basic_model.parameters(), 
                                                 lr=cond_train['lr_basic_model'])
        ##########################在该初始化下，交替进行图优化与模型优化#######################
        loss_avg = 0
        for step_syn in range(cond_train['epochs_deep']):
            basic_model.eval()# fix basic_model while optimizing graphsyner
            #------------------------------------------------------------------
            x_syn_dict = graphsyner.get_x_syn_dict()
            adj_t_syn_dict = graphsyner.get_adj_t_syn_dict()
            ######################先进行图优化##################################### cross_entropy()
            output = basic_model(x_dict, adj_t_dict)
            loss_real = F.nll_loss(output[train_mask], y[train_mask])
            gw_reals = torch.autograd.grad(loss_real, parameters)
            gw_reals = list((_.detach().clone() for _ in gw_reals)) 
            #real gradient
            #------------------------------------------------------------------
            output_syn = basic_model(x_syn_dict, adj_t_syn_dict)[mask_syn]
            loss_syn = F.nll_loss(output_syn, y_syn[mask_syn])
            gw_syns = torch.autograd.grad(loss_syn, parameters, create_graph=True)
            #synthetic gadient
            #------------------------------------------------------------------
            loss = match_loss(gw_syns, gw_reals)
            optimizer_cond.zero_grad()
            loss.backward()
            optimizer_cond.step()
            #------------------------------------------------------------------
            loss_avg += loss.item()/cond_train['epochs_deep']
            #########################再进行模型参数优化###########################
            if step_syn < cond_train['epochs_deep']-1:
                x_syn_dict = graphsyner.get_x_syn_detached_dict()
                adj_t_syn_dict = graphsyner.get_adj_t_syn_detached_dict()
                train_model(basic_model, cond_train, optimizer_basic_model, 
                            x_syn_dict, adj_t_syn_dict, y_syn, mask_syn)
        losses_log.append(loss_avg)
        if loss_avg < smallest_loss:
            smallest_loss = loss_avg
            graphsyner.best_x_syn_dict = graphsyner.get_x_syn_detached_dict()
            graphsyner.best_adj_t_syn_dict = graphsyner.get_adj_t_syn_detached_dict()
    return graphsyner, losses_log
