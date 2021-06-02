import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MLP(torch.nn.Module):
    def __init__(self, in_channels,n_freq_bands, hidden_channels, n_classes, dr):
        super(MLP, self).__init__()

        self.dr = dr

        self.lin1 = torch.nn.Linear(n_freq_bands*32*in_channels, hidden_channels*4)
        self.lin2 = torch.nn.Linear(hidden_channels*4, hidden_channels*2)
        self.lin3 = torch.nn.Linear(hidden_channels*2, hidden_channels)
        self.lin4 = torch.nn.Linear(hidden_channels, n_classes)

        self.act = nn.Softmax(dim=-1)


    def forward(self, batch):
        bs = len(torch.unique(batch.batch))
        x = batch.x

        x = rearrange(x, '(bs g e) f -> bs (e g f)', bs=bs, e=32)

        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dr/2, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        x = x.relu()
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.lin4(x)
        x = self.act(x)
        return x