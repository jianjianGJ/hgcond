#%%
import torch
import torch_geometric.transforms as T 
from torch_geometric.datasets import DBLP, OGB_MAG, IMDB, HGBDataset, AMiner
from torch_geometric.datasets.fake import FakeHeteroDataset
from utils import asymmetric_gcn_norm

import prettytable as pt

#%%
def get_AMiner(root):
    dataset = AMiner(root+'/AMiner', transform=T.ToSparseTensor(remove_edge_index=False))
    data = dataset[0]
    data['author'].x = torch.load(root+'/AMiner/raw/author.tensor')
    data['paper'].x = torch.load(root+'/AMiner/raw/paper.tensor')
    data['venue'].x = torch.load(root+'/AMiner/raw/venue.tensor')
    #________________________train/val/test split__________________________________
    # torch.manual_seed(0)
    # index_labeled_rand = data['author'].y_index[torch.randperm(data['author'].y_index.shape[0])]
    # torch.save(index_labeled_rand,root+'/AMiner/raw/index_labeled_rand.tensor')
    
    y = torch.ones(data['author'].x.shape[0], dtype=torch.long)*-1
    y[data['author'].y_index] = data['author'].y
    data['author'].y = y
    index_labeled_rand = torch.load(root+'/AMiner/raw/index_labeled_rand.tensor')
    n = index_labeled_rand.shape[0]
    index_train = index_labeled_rand[:int(n*0.3)]
    index_val = index_labeled_rand[int(n*0.3):int(n*0.6)]
    index_test = index_labeled_rand[int(n*0.6):]
    data['author'].train_mask = torch.zeros(data['author'].x.shape[0], dtype=torch.bool)
    data['author'].train_mask[index_train] = True
    data['author'].val_mask = torch.zeros(data['author'].x.shape[0], dtype=torch.bool)
    data['author'].val_mask[index_val] = True
    data['author'].test_mask = torch.zeros(data['author'].x.shape[0], dtype=torch.bool)
    data['author'].test_mask[index_test] = True
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    for edge_type in data.edge_types:
        data[edge_type]['adj_t'] = asymmetric_gcn_norm(data[edge_type]['adj_t'])
    data.target_node_type='author'
    data.num_classes = int(data['author'].y.max()+1)
    data.name = 'AMiner'
    node_info, edge_info = get_datainfo(data)
    data.info = node_info+'\n'+edge_info
    return data
def get_DBLP(root):
    dataset = DBLP(root+'/DBLP', transform=T.ToSparseTensor(remove_edge_index=False))
    data = dataset[0]
    data['conference'].x = torch.ones(data['conference'].num_nodes, 1)
    for edge_type in data.edge_types:
        data[edge_type]['adj_t'] = asymmetric_gcn_norm(data[edge_type]['adj_t'])
    data.target_node_type='author'
    data.num_classes = int(data['author'].y.max()+1)
    data.name = 'dblp'
    node_info, edge_info = get_datainfo(data)
    data.info = node_info+'\n'+edge_info
    return data

def get_IMDB(root):
    dataset = IMDB(root+'/IMDB', transform=T.ToSparseTensor(remove_edge_index=False))
    data = dataset[0]
    for edge_type in data.edge_types:
        data[edge_type]['adj_t'] = asymmetric_gcn_norm(data[edge_type]['adj_t'])
    data.target_node_type='movie'
    data.num_classes = int(data['movie'].y.max()+1)
    data.name = 'imdb'
    node_info, edge_info = get_datainfo(data)
    data.info = node_info+'\n'+edge_info
    return data


def get_ACM(root):
    dataset = HGBDataset(root, name='ACM',transform=T.ToSparseTensor(remove_edge_index=False))
    data = dataset[0]
    data['term'].x = torch.ones(data['term'].num_nodes, 1)
    for edge_type in data.edge_types:
        data[edge_type]['adj_t'] = asymmetric_gcn_norm(data[edge_type]['adj_t'])
    data.target_node_type='paper'
    data.num_classes = int(data['paper'].y.max()+1)
    data.name = 'acm'
    #--------------------------------------------------------------------------
    # for reproduce, we fix the splits
    # torch.manual_seed(0)
    # index = torch.arange(data['paper'].y.shape[0])
    # index_labeled = index[data['paper'].train_mask]
    # index_labeled_rand = index_labeled[torch.randperm(index_labeled.shape[0])]
    index_labeled_rand = torch.load(root+'/acm/raw/index_labeled_rand.tensor')
    index_train = index_labeled_rand[:200]
    index_val = index_labeled_rand[200:400]
    index_test = index_labeled_rand[400:]
    data['paper'].train_mask = torch.zeros(data['paper'].y.shape[0], dtype=torch.bool)
    data['paper'].train_mask[index_train] = True
    data['paper'].val_mask = torch.zeros(data['paper'].y.shape[0], dtype=torch.bool)
    data['paper'].val_mask[index_val] = True
    data['paper'].test_mask = torch.zeros(data['paper'].y.shape[0], dtype=torch.bool)
    data['paper'].test_mask[index_test] = True
    #--------------------------------------------------------------------------
    node_info, edge_info = get_datainfo(data)
    data.info = node_info+'\n'+edge_info
    return data
def get_Freebase(root):
    transform = T.Compose([T.ToUndirected(), T.ToSparseTensor(remove_edge_index=False)])#
    dataset = HGBDataset(root, name='Freebase',transform=transform)
    data = dataset[0]
    ######################################################### set features
    x_dict = torch.load('../datahetero/freebase/raw/x_dict.dict')
    for node_type in data.node_types:
        data[node_type].x = x_dict[node_type]
    ######################################################### set features done
    for edge_type in data.edge_types:
        data[edge_type]['adj_t'] = asymmetric_gcn_norm(data[edge_type]['adj_t'])
    data.target_node_type='book'
    data.num_classes = int(data['book'].y.max()+1)
    data.name = 'freebase'
    #-------------------------------------------------------------total 2386 labels
    # torch.manual_seed(0)
    # index = torch.arange(data['book'].y.shape[0])
    # index_labeled = index[data['book'].train_mask]
    # index_labeled_rand = index_labeled[torch.randperm(index_labeled.shape[0])]
    # torch.save(index_labeled_rand, root+'/freebase/raw/index_labeled_rand.tensor')
    # for reproduce, we fix the splits
    index_labeled_rand = torch.load(root+'/freebase/raw/index_labeled_rand.tensor')
    index_train = index_labeled_rand[:500]
    index_val = index_labeled_rand[500:1000]
    index_test = index_labeled_rand[1000:]
    data['book'].train_mask = torch.zeros(data['book'].y.shape[0], dtype=torch.bool)
    data['book'].train_mask[index_train] = True
    data['book'].val_mask = torch.zeros(data['book'].y.shape[0], dtype=torch.bool)
    data['book'].val_mask[index_val] = True
    data['book'].test_mask = torch.zeros(data['book'].y.shape[0], dtype=torch.bool)
    data['book'].test_mask[index_test] = True
    #--------------------------------------------------------------------------
    node_info, edge_info = get_datainfo(data)
    data.info = node_info+'\n'+edge_info
    return data

# dblp imdb acm mag
def get_data(name, root = '../datahetero', ):
    if name.lower() == 'dblp':
        return get_DBLP(root)
    elif name.lower() == 'imdb':
        return get_IMDB(root)
    elif name.lower() == 'acm':
        return get_ACM(root)
    elif name.lower() == 'freebase':
        return get_Freebase(root)
    elif name.lower() == 'aminer':
        return get_AMiner(root)
    else:
        raise NotImplementedError
def get_datainfo(data):
    node_table = pt.PrettyTable(['TYPE', 'NUM', 'CHANNEL','CLASS', 'SPLIT'])
    node_types = data.node_types
    target_node_type = data.target_node_type
    for node_type in node_types:
        info_dict = data[node_type]
        TYPE = node_type
        NUM, CHANNEL = info_dict['x'].shape[0], info_dict['x'].shape[1]
        CLASS, SPLIT = '', ''
        if node_type==target_node_type:
            CLASS = data.num_classes
            if 'train_mask' in info_dict.keys():
                train_mask = info_dict['train_mask']
                num_train = train_mask.sum().item()
            else:
                num_train = '-'
            if 'val_mask' in info_dict.keys():
                val_mask = info_dict['val_mask']
                num_val = val_mask.sum().item()
            else:
                num_val = '-'
            if 'test_mask' in info_dict.keys():
                test_mask = info_dict['test_mask']
                num_test = test_mask.sum().item()
            else:
                num_test = '-'
            SPLIT = f'{num_train}/{num_val}/{num_test}'
        node_table.add_row([TYPE, NUM, CHANNEL, CLASS, SPLIT])
    node_table.title = f'Node info of {data.name.upper()}'
    node_info = node_table.get_string()  
    edge_table = pt.PrettyTable(['TYPE', 'NUM', 'SHAPE','SPARSITY'])
    edge_types = data.edge_types
    for edge_type in edge_types:
        adj_t = data[edge_type]['adj_t']
        TYPE = edge_type
        NUM = adj_t.nnz()
        SHAPE = f'({adj_t.size(0)},{adj_t.size(1)})'
        SPARSITY = f'{NUM/adj_t.size(0)/adj_t.size(1)*100:.2f}%'
        edge_table.add_row([TYPE, NUM, SHAPE, SPARSITY])
    edge_table.title = f'Edge info of {data.name.upper()}'
    edge_info = edge_table.get_string()  
    return node_info, edge_info
if __name__ == '__main__':
    datanames = ['dblp', 'imdb', 'acm', 'aminer', 'freebase'] 
    info = ''
    for dataname in datanames:
        print(dataname)
        data = get_data(dataname)
        node_info, edge_info = get_datainfo(data)
        info += node_info
        info += '\n'
        info += edge_info
        info += '\n'
        info += '\n'
    with open('datainfo.txt','w') as f:    #设置文件对象
        f.write(info) 
        
        
    





