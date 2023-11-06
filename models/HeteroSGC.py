import torch
from torch_geometric.nn import Linear
import torch.nn.functional as F

class HeteroSGC(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers,
                  node_types, edge_types, target_node_type, num_lins=1, alpha=0.01):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.alpha = alpha
        self.num_layers = num_layers
        self.num_lins = num_lins
        self.target_node_type = target_node_type
        
        self.in_lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.in_lin_dict[node_type] = torch.nn.ModuleList()
            self.in_lin_dict[node_type].append(Linear(-1, hidden_channels))
            for _ in range(num_lins-1):
                self.in_lin_dict[node_type].append(Linear(hidden_channels, hidden_channels))
        self.out_lin = Linear(hidden_channels, out_channels)
    def reset_parameters(self,init_list=None):
        if init_list is None:
            for node_type in self.in_lin_dict.keys():
                for lin in self.in_lin_dict[node_type]:
                    lin.reset_parameters()
            self.out_lin.reset_parameters()
        else:
            i = 0
            for self_p in self.parameters():
                if self_p.dim()==2:
                    self_p.data.copy_(init_list[i])
                    i += 1
            
    def forward(self, x_dict, adj_t_dict, get_embeddings=False):
        h_dict = {}
        for node_type in x_dict.keys():
            h_dict[node_type] = self.in_lin_dict[node_type][0](x_dict[node_type]).relu_()
            for lin in self.in_lin_dict[node_type][1:]:
                h_dict[node_type] = lin(h_dict[node_type]).relu_()
        for l in range(self.num_layers):
            out_dict = {node_type: [self.alpha*x] for node_type,x in h_dict.items()}
            # out_dict = {node_type: [] for node_type,x in h_dict.items()}
            for edge_type, adj_t in adj_t_dict.items():
                src_type, _, dst_type = edge_type
                out_dict[dst_type].append(adj_t @ h_dict[src_type])
            for node_type in x_dict.keys():
                h_dict[node_type] = torch.sum(torch.stack(out_dict[node_type],dim=0), dim=0)
        target_logits = self.out_lin(h_dict[self.target_node_type])
        if get_embeddings:
            h_dict = {node_type:h for node_type,h in h_dict.items()}
            return target_logits, h_dict
        else:
            return target_logits

