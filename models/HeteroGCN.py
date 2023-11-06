import torch
from torch_geometric.nn import Linear
import torch.nn.functional as F

# class GCNconv(torch.nn.Module):
#     def __init__(self, hidden_channels, alpha = 0.01):
#         super().__init__()
#         self.alpha = alpha
#         self.lin_src = Linear(-1, hidden_channels, bias=False)
#         self.lin_dst = Linear(-1, hidden_channels, bias=False)
#     def reset_parameters(self):
#         self.lin_src.reset_parameters()
#         self.lin_dst.reset_parameters()
#     def forward(self, x_tuple, adj_t):
#         x_src, x_dst = x_tuple
#         h_src, h_dst = self.lin_src(x_src), self.lin_dst(x_dst)
#         out_dst = adj_t @ h_src + self.alpha*h_dst
#         # out_dst = self.lin_src(adj_t @ x_src) + self.alpha*self.lin_dst(x_dst)
#         return out_dst
    
# class HeteroGCN(torch.nn.Module):
#     def __init__(self, hidden_channels, out_channels, num_layers,
#                   edge_types, node_types, target_node_type):
#         super().__init__()
#         self.num_layers = num_layers
#         self.node_types = node_types
#         self.edge_types = edge_types
#         self.target_node_type = target_node_type
#         self.convs = torch.nn.ModuleList()
#         for l in range(num_layers):
#             conv = torch.nn.ModuleDict()
#             if l==self.num_layers-1: # the last layer
#                 for edge_type in edge_types:
#                     edge_type_str = '_'.join(edge_type)
#                     src_type, _, dst_type = edge_type
#                     if dst_type == self.target_node_type:
#                         conv[edge_type_str] = GCNconv(hidden_channels)
#             else:
#                 for edge_type in edge_types:
#                     edge_type_str = '_'.join(edge_type)
#                     conv[edge_type_str] = GCNconv(hidden_channels)
#             self.convs.append(conv)
#         self.out_lin = Linear(hidden_channels, out_channels)
#     def reset_parameters(self,init_list=None):
#         if init_list is None:
#             for conv in self.convs:
#                 for conv_ in conv.values():
#                     conv_.reset_parameters()
#             self.out_lin.reset_parameters()
#         else:
#             i = 0
#             for self_p in self.parameters():
#                 if self_p.dim()==2:
#                     self_p.data.copy_(init_list[i])
#                     i += 1
#     def forward(self, x_dict, adj_t_dict, get_embeddings=False):
#         h_dict = {node_type:x.clone() for node_type,x in x_dict.items()}
#         for l in range(self.num_layers):
#             conv = self.convs[l]
#             out_dict = {node_type: [] for node_type in h_dict.keys()}
#             for edge_type in self.edge_types:
#                 src_type, _, dst_type = edge_type
#                 edge_type_str = '_'.join(edge_type)
#                 if edge_type_str in conv.keys():
#                     out_dict[dst_type].append(conv[edge_type_str]((h_dict[src_type], h_dict[dst_type]), adj_t_dict[edge_type]))
#             for node_type in x_dict.keys():
#                 if len(out_dict[node_type])>0:
#                     h_dict[node_type] = torch.sum(torch.stack(out_dict[node_type],dim=0), dim=0).relu_()
#         target_logits = self.out_lin(h_dict[self.target_node_type])
#         if get_embeddings:
#             h_dict = {node_type:h for node_type,h in h_dict.items()}
#             return target_logits, h_dict
#         else:
#             return target_logits
        
        
# class HeteroGCN(torch.nn.Module):# batch norm
#     def __init__(self, hidden_channels, out_channels, num_layers,
#                   edge_types, node_types, target_node_type):
#         super().__init__()
#         self.num_layers = num_layers
#         self.node_types = node_types
#         self.edge_types = edge_types
#         self.target_node_type = target_node_type
#         self.convs = torch.nn.ModuleList()
#         self.bns = torch.nn.ModuleList()
#         for l in range(num_layers):
#             conv = torch.nn.ModuleDict()
#             bn = torch.nn.ModuleDict()
#             for node_type in self.node_types:
#                 bn[node_type] = torch.nn.BatchNorm1d(hidden_channels)
#             self.bns.append(bn)
#             if l==self.num_layers-1:
#                 for edge_type in edge_types:
#                     edge_type_str = '_'.join(edge_type)
#                     src_type, _, dst_type = edge_type
#                     if dst_type == self.target_node_type:
#                         conv[edge_type_str] = GCNconv(hidden_channels)
#             else:
#                 for edge_type in edge_types:
#                     edge_type_str = '_'.join(edge_type)
#                     conv[edge_type_str] = GCNconv(hidden_channels)
#             self.convs.append(conv)
#         self.out_lin = Linear(hidden_channels, out_channels)
#     def reset_parameters(self,init_list=None):
#         if init_list is None:
#             for conv in self.convs:
#                 for conv_ in conv.values():
#                     conv_.reset_parameters()
#             self.out_lin.reset_parameters()
#         else:
#             i = 0
#             for self_p in self.parameters():
#                 if self_p.dim()==2:
#                     self_p.data.copy_(init_list[i])
#                     i += 1
#     def forward(self, x_dict, adj_t_dict, get_embeddings=False):
#         h_dict = {node_type:x.clone() for node_type,x in x_dict.items()}
#         for l in range(self.num_layers):
#             conv = self.convs[l]
#             out_dict = {node_type: [] for node_type in h_dict.keys()}
#             for edge_type in self.edge_types:
#                 src_type, _, dst_type = edge_type
#                 edge_type_str = '_'.join(edge_type)
#                 if edge_type_str in conv.keys():
#                     out_dict[dst_type].append(self.bns[l][dst_type](conv[edge_type_str]((h_dict[src_type], h_dict[dst_type]), adj_t_dict[edge_type])))
#             for node_type in x_dict.keys():
#                 if len(out_dict[node_type])>0:
#                     h_dict[node_type] = torch.sum(torch.stack(out_dict[node_type],dim=0), dim=0).relu_()
#         target_logits = self.out_lin(h_dict[self.target_node_type])
#         if get_embeddings:
#             h_dict = {node_type:h for node_type,h in h_dict.items()}
#             return target_logits, h_dict
#         else:
#             return target_logits



#------------------------------------------------------------------------------
# In a layer, a node_type has noly one Linear, no matter edge_type
#------------------------------------------------------------------------------
class HeteroGCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers,
                  edge_types, node_types, target_node_type, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.num_layers = num_layers
        self.target_node_type = target_node_type
        
        self.lins = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lins[node_type] = torch.nn.ModuleList()
            self.lins[node_type].append(Linear(-1, hidden_channels,bias=False))
            for _ in range(num_layers-1):
                self.lins[node_type].append(Linear(hidden_channels, hidden_channels,bias=False))
        self.out_lin = Linear(hidden_channels, out_channels)
    def reset_parameters(self,init_list=None):
        if init_list is None:
            for node_type in self.lins.keys():
                for lin in self.lins[node_type]:
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
        for l in range(self.num_layers):
            for node_type in x_dict.keys():
                if l==0:
                    h_dict[node_type] = F.relu_(self.lins[node_type][l](x_dict[node_type]))
                else:
                    h_dict[node_type] = F.relu_(self.lins[node_type][l](h_dict[node_type]))
            out_dict = {node_type: [self.alpha*x] for node_type,x in h_dict.items()}#
            for edge_type, adj_t in adj_t_dict.items():
                src_type, _, dst_type = edge_type
                out_dict[dst_type].append(adj_t @ h_dict[src_type])
            for node_type in x_dict.keys():
                h_dict[node_type] = torch.mean(torch.stack(out_dict[node_type],dim=0), dim=0)#.relu_()
        
        target_logits = self.out_lin(h_dict[self.target_node_type])
        if get_embeddings:
            h_dict = {node_type:h for node_type,h in h_dict.items()}
            return target_logits, h_dict
        else:
            return target_logits
