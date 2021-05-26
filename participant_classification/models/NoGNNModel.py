import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from einops import rearrange

class NoGNNModel(torch.nn.Module):
    def __init__(self, in_channels,n_graphs, hidden_channels, n_classes):
        super(NoGNNModel, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        self.cnn1 = torch.nn.Conv1d(n_graphs, 1, kernel_size=3, stride=2)
        
        self.lin1 = torch.nn.Linear(32*(in_channels//2 - (1 if in_channels%2 == 0 else 0)), hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, n_classes)

        self.softmax = nn.Softmax(dim=-1)

        
    def forward(self, batch):
        bs = len(torch.unique(batch.batch))
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        # x = self.gconv1(x, edge_index, edge_attr)
        # x = self.gconv2(x, edge_index, edge_attr)
        # x = F.dropout(x, p=0.4, training=self.training)
        # x = x.relu()
        x = rearrange(x, '(bs g e) f -> (bs e) g f', bs=bs, e=32)
        x = self.cnn1(x).squeeze()
        x = x.tanh()
        x = rearrange(x, '(bs e) f -> bs (e f)', bs=bs)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = self.softmax(x)
        return x