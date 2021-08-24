import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, dr, activation_fn):
        super(MLP, self).__init__()

        self.lin1 = torch.nn.Linear(32*in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, n_classes)

        self.act = nn.ReLU() if activation_fn == 'relu' else nn.Tanh()
        self.final_act = nn.Softmax(dim=-1)

        self.dr = dr

    def forward(self, batch):
        x = batch.x
        bs = len(torch.unique(batch.batch)) if 'batch' in dir(batch) else 1
        x = rearrange(x, '(bs e) f -> bs (e f)', bs=bs)

        x = self.lin1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.lin2(x)
        x = self.final_act(x)
        return x
