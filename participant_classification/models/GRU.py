import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GRU(torch.nn.Module):
    def __init__(self, in_channels,n_freq_bands, hidden_channels, n_classes, dr):
        super(GRU, self).__init__()

        self.dr = dr
        self.hidden_channels = hidden_channels

        self.grus = nn.ModuleList([nn.GRU(32, hidden_channels, 2, batch_first = True) for _ in range(n_freq_bands)])

        self.lin1 = torch.nn.Linear(n_freq_bands*hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, n_classes)

        self.act = nn.Softmax(dim=-1)


    def forward(self, batch):
        bs = len(torch.unique(batch.batch))
        x = batch.x

        print(x.shape)

        x = rearrange(x, '(bs g e) f -> g bs f e', bs=bs, e=32)

        print(x.shape)
        exit()

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