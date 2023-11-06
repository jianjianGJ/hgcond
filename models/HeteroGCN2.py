import torch
from torch_geometric.nn import Linear
import torch.nn.functional as F

#------------------------------------------------------------------------------
# In a layer, a node_type has noly one Linear, no matter edge_type
#------------------------------------------------------------------------------
class HeteroGCN2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers,
                  edge_types, node_types, target_node_type, alpha=1, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
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
        h0_dict = {}
        for l in range(self.num_layers):
            for node_type in x_dict.keys():
                if l==0:
                    h_dict[node_type] = F.relu_(self.lins[node_type][l](x_dict[node_type]))
                    h0_dict[node_type] = h_dict[node_type].detach()
                else:
                    h_dict[node_type] = F.relu_(self.lins[node_type][l](h_dict[node_type]))
            out_dict = {node_type: [self.alpha*x] for node_type,x in h_dict.items()}#
            for edge_type, adj_t in adj_t_dict.items():
                src_type, _, dst_type = edge_type
                out_dict[dst_type].append(adj_t @ h_dict[src_type])
            for node_type in x_dict.keys():
                h_dict[node_type] = self.beta*h0_dict[node_type] + (1-self.beta)*torch.mean(torch.stack(out_dict[node_type],dim=0), dim=0)#.relu_()
        
        target_logits = self.out_lin(h_dict[self.target_node_type])
        if get_embeddings:
            h_dict = {node_type:h for node_type,h in h_dict.items()}
            return target_logits, h_dict
        else:
            return target_logits
