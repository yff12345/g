import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool as gap
from einops import rearrange


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, dr, activation_fn):
        super(GIN, self).__init__()
        
        self.gconv1 = GINConv(nn.Linear(in_channels, in_channels))
        
        self.lin1 = nn.Linear(in_channels*32,hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, n_classes)

        self.act = nn.ReLU() if activation_fn == 'relu' else nn.Tanh()
        self.final_act = nn.Softmax(dim=-1)

        self.dr = dr
        
        
    def forward(self, batch):
        bs = len(torch.unique(batch.batch)) if 'batch' in dir(batch) else 1
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x = self.gconv1(x, edge_index)
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