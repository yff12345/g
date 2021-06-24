import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import GCNConv

class GCNGRU(torch.nn.Module):
    def __init__(self, in_channels,n_freq_bands, hidden_channels, n_classes, dr):
        super(GCNGRU, self).__init__()

        self.dr = dr
        self.hidden_channels = hidden_channels

        self.grus = nn.ModuleList([nn.GRU(32, hidden_channels, 2, batch_first = True) for _ in range(n_freq_bands)])

        self.gconv1 = GCNConv(in_channels,hidden_channels*2)
        self.gconv2 = GCNConv(hidden_channels*2,hidden_channels)

        self.lin1 = nn.Linear(n_freq_bands*hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, n_classes)

        self.act = nn.Softmax(dim=-1)


    def forward(self, batch):
        bs = len(torch.unique(batch.batch)) if 'batch' in dir(batch) else 1
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x = self.gconv1(x, edge_index, edge_attr)
        x = self.gconv2(x, edge_index, edge_attr)
    
        x = rearrange(x, '(bs g e) f -> g bs f e', bs=bs, e=32)

        xs = []
        for i, freq_band_x in enumerate(x):
            out, h_n = self.grus[i](freq_band_x)
            xs.append(h_n[-1])

        x = torch.stack(xs)
        x = rearrange(x, 'g bs hc -> bs (g hc)')
        x = F.dropout(x, p=self.dr/2, training=self.training)

        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.lin2(x)
        x = self.act(x)
        return x