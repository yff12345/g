import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv as nnGraphConv, global_add_pool as gap
from einops import rearrange


class GraphConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, dr, activation_fn):
        super(GraphConv, self).__init__()
        
        self.gconv1 = nnGraphConv(in_channels,hidden_channels)
        self.gconv2 = nnGraphConv(hidden_channels,hidden_channels)
        
        self.lin1 = torch.nn.Linear(hidden_channels*32,hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, n_classes)

        self.act = nn.ReLU() if activation_fn == 'relu' else nn.Tanh()
        self.final_act = nn.Softmax(dim=-1)

        self.dr = dr
        
        
    def forward(self, batch):
        bs = len(torch.unique(batch.batch)) if 'batch' in dir(batch) else 1
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x = self.gconv1(x, edge_index, edge_attr)
        x = self.act(x)
        x = F.dropout(x, p=self.dr/2, training=self.training)
        x = self.gconv2(x, edge_index, edge_attr)
        x = self.act(x)

        # Flatten
        x = rearrange(x, '(bs e) f -> bs (f e)', bs=bs)
        # Global add pooling
        # x = gap(x,batch.batch)

        x = self.lin1(x)
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.act(x)
        x = self.lin2(x)
        x = self.final_act(x)
        return x