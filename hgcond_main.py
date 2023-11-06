import os
import argparse
import time
import numpy as np
import torch
from graphsyn import evalue_hgcond, hgcond
from utils_data import get_data
from utils import getsize_mb
#%% cmd parameters setting
argparser = argparse.ArgumentParser("HGCond",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--dataset", type=str, default='imdb')#dblp
argparser.add_argument("--cond", type=float, default=0.001)
argparser.add_argument("--basicmodel", type=str, default='HeteroSGC')#HeteroSGC HeteroGCN
argparser.add_argument("--para-init", type=str, default='orth')# 'orth' 'rand'
argparser.add_argument("--feat-init", type=str, default='cluster') # 'cluster' 'sample'
args = argparser.parse_args()
dataset = args.dataset#['dblp', 'imdb', 'acm', 'AMiner','freebase']
cond_rate = args.cond
basicmodel = args.basicmodel
feat_init = args.feat_init
para_init = args.para_init
#%% fix parameters setting
model_architecture = {'hidden_channels':64,
                      'num_layers':3}
model_train = {'epochs':1000,
                'lr':0.01,
                'weight_decay':0.0005}
cond_train = {'epochs_deep':3,
                'epochs_basic_model':1,
                'lr':0.01,
                'lr_basic_model':0.1}
cond_train['epochs_initial'] = model_architecture['hidden_channels']*2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#%% Necessary directory
dir_name = './synthetic_graphs'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
#%% Data
data = get_data(name=dataset)
data = data.to(device)

#%% Train
info = f'Dataset:{dataset}\nCond-rate:{cond_rate}\nFeature initialization:{feat_init}\nParameter initialization:{para_init}\nBasicmodel:{basicmodel}\n'
print(info)
print('Original dataset infomations:')
print(data.info)
print('Start condensing ...')
time_start=time.time()
# graphsyner, losses_log = hgcond(data, cond_rate, feat_init, para_init, 
#                                 basicmodel, model_architecture, cond_train)
####################################################
graphsyner, losses_log = hgcond(data, cond_rate, feat_init, para_init, basicmodel, model_architecture, cond_train)
#################################################
time_end=time.time()
y_syn = graphsyner.y_syn
mask_syn = graphsyner.mask_syn
x_syn_dict = graphsyner.best_x_syn_dict
adj_t_syn_dict = graphsyner.best_adj_t_syn_dict
save_path = f'{dir_name}/{dataset}-{cond_rate}.cond'
torch.save((x_syn_dict, adj_t_syn_dict, y_syn, mask_syn), save_path)
print(f'Condensation finished, taking time:{time_end - time_start:.2f}s')
print(f'The condensed graph is saved as {save_path}')
print(graphsyner)
origin_storage = getsize_mb([data.x_dict, data.adj_t_dict, 
                             data[data.target_node_type].y,
                             data[data.target_node_type].train_mask])
condensed_storage = getsize_mb([x_syn_dict, adj_t_syn_dict, y_syn, mask_syn])
print(f'Origin graph:{origin_storage:.2f}Mb  Condensed graph:{condensed_storage:.2f}Mb')
    #%%
print('Train on the synthetic graph and test on the real graph')
# x_syn_dict, adj_t_syn_dict, y_syn, mask_syn = torch.load('./synthetic_graphs/dblp-0.001-cluster-orth-1.cond')
accs,f1_micros,f1_macros = evalue_hgcond(1, data, x_syn_dict, adj_t_syn_dict, y_syn, mask_syn,
                      basicmodel, model_architecture, model_train)
mean = np.mean(accs)*100
std = np.std(accs)*100
print(f'\nAccuracy:{mean:.2f}+{std:.2f}')





