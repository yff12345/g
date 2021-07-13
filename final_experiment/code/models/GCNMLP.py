import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from einops import rearrange

class GCNMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, dr):
        super(GCNMLP, self).__init__()

        self.dr = dr

        self.gconv1 = GCNConv(in_channels,hidden_channels)
        # self.gconv2 = GCNConv(hidden_channels*2,hidden_channels)

        self.lin1 = torch.nn.Linear(32*hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels//2)
        self.lin3 = torch.nn.Linear(hidden_channels//2, n_classes)

        self.act = nn.Softmax(dim=-1)


    def forward(self, batch):
        bs = len(torch.unique(batch.batch)) if 'batch' in dir(batch) else 1
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        
        x = self.gconv1(x, edge_index, edge_attr)
        # x = self.gconv2(x, edge_index, edge_attr)

        x = rearrange(x, '(bs e) f -> bs (e f)', bs=bs)

        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dr/2, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.lin3(x)
        x = self.act(x)
        return x