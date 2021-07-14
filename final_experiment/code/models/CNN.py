import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, dr, activation_fn):
        super(CNN, self).__init__()
        
        self.cnn1 = torch.nn.Conv1d(4, 1, kernel_size=1, stride=1)
        
        self.lin1 = torch.nn.Linear(32, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels*2)
        self.lin3 = torch.nn.Linear(hidden_channels*2, hidden_channels)
        self.lin4 = torch.nn.Linear(hidden_channels, n_classes)

        self.act = nn.ReLU() if activation_fn == 'relu' else nn.Tanh()
        self.final_act = nn.Softmax(dim=-1)

        self.dr = dr
        
        
    def forward(self, batch):
        bs = len(torch.unique(batch.batch)) if 'batch' in dir(batch) else 1
        x = batch.x
        
        x = rearrange(x, '(bs e) f -> bs f e', bs=bs)
        x = self.cnn1(x).squeeze()
        x = self.act(x)
        x = F.dropout(x, p=self.dr/2, training=self.training)
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        x = self.act(x)
        x = self.lin3(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.lin4(x)
        x = self.final_act(x)
        return x