import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CNN(torch.nn.Module):
    def __init__(self, in_channels,n_graphs, hidden_channels, n_classes, dr):
        super(CNN, self).__init__()
        
        self.dr = dr
        
        self.cnn1 = torch.nn.Conv1d(n_graphs, 1, kernel_size=1, stride=1)
        
        # print('#aaaaaaaaaaaaaaaaaaaa')
        # print(32*(in_channels//2 - (1 if in_channels%2 == 0 else 0)))
        self.lin1 = torch.nn.Linear(32*in_channels, hidden_channels)
        # print(';opk--------------------')
        self.lin2 = torch.nn.Linear(hidden_channels, n_classes)

        self.act = nn.Softmax(dim=-1)
        
        
    def forward(self, batch):
        bs = len(torch.unique(batch.batch)) if 'batch' in dir(batch) else 1
        x = batch.x
        x = rearrange(x, '(bs g e) f -> (bs e) g f', bs=bs, e=32)
        x = self.cnn1(x).squeeze()
        x = x.tanh()
        x = F.dropout(x, p=self.dr/2, training=self.training)
        x = rearrange(x, '(bs e) f -> bs (e f)', bs=bs)
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.lin2(x)
        x = self.act(x)
        return x