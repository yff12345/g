import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, dr):
        super(MLP, self).__init__()

        self.dr = dr

        self.lin1 = torch.nn.Linear(32*in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels//2)
        self.lin3 = torch.nn.Linear(hidden_channels//2, hidden_channels//4)
        self.lin4 = torch.nn.Linear(hidden_channels//4, n_classes)

        self.act = nn.Softmax(dim=-1)


    def forward(self, batch):
        x = batch.x
        bs = len(torch.unique(batch.batch)) if 'batch' in dir(batch) else 1

        x = rearrange(x, '(bs e) f -> bs (e f)', bs=bs)

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