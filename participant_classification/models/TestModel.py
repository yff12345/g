import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GatedGraphConv
from einops import rearrange

class TestModel(torch.nn.Module):
    def __init__(self, in_channels,n_graphs, hidden_channels, n_classes, dr):
        super(TestModel, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        self.gconv1 = GraphConv(in_channels,2)
        self.gconv2 = GraphConv(2, 1)

        self.grus = nn.GRU(32, hidden_channels, 1, batch_first = True)
        
        self.cnn1 = torch.nn.Conv1d(n_graphs, 1, kernel_size=5, stride=1)
        
        # self.lin1 = torch.nn.Linear(32*(hidden_channels//2 - (1 if hidden_channels%2 == 0 else 0)), hidden_channels)
        # self.lin1 = torch.nn.Linear(65408, hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels*4)
        self.lin2 = torch.nn.Linear(hidden_channels*4, n_classes)
        self.act = nn.Softmax(dim=-1)

        self.dr = dr

        
    def forward(self, batch):
        bs = len(torch.unique(batch.batch))
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        # print(x.shape)

        # x = rearrange(x, '(bs g e) f -> g bs f e', bs=bs, e=32)

        # 
        
        x = self.gconv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.gconv2(x, edge_index, edge_attr)
        x = x.relu()
        x = F.dropout(x, p=self.dr/2, training=self.training)

        x = rearrange(x, '(bs g e) f -> bs g (e f)', bs=bs, e=32)

        out, h_n = self.grus(x)

        x = h_n[-1]

        x = self.lin1(x)
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.act(x)
        return x
        