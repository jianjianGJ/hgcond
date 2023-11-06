#https://arxiv.org/pdf/1903.07293.pdf
import torch
from torch import nn

from torch_geometric.nn import HANConv, Linear



class HAN(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, 
                 edge_types, node_types, target_node_type):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.target_node_type = target_node_type
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HANConv(hidden_channels, hidden_channels, heads=num_heads,
                                dropout=0.5, metadata=(node_types,edge_types))
            self.convs.append(conv)
        self.lin = nn.Linear(hidden_channels, out_channels)
    def reset_parameters(self):
        for node_type in self.node_types:
            self.lin_dict[node_type].reset_parameters()
        self.lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict[self.target_node_type])

