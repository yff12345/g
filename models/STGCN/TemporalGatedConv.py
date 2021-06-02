import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

class TemporalGatedConv(torch.nn.Module):
    def __init__(self , in_channels, out_channels, kernel_size,stride=1, dr = 0):
        super(TemporalGatedConv, self).__init__()
        self.conv = nn.Conv1d(in_channels , 2*out_channels, kernel_size, stride)
        self.dropout = nn.Dropout(dr, inplace=True)

    def forward(self, x_in):
        x = rearrange(x_in,'N M n c -> (N n) c M')
        x = self.conv(x)
        x = F.relu(x)
        x = F.glu(x,dim=1)
        x = rearrange(x,'(N n) c M -> N M n c',N=x_in.shape[0])
        # TODO: Add residual connection, also dropout
        return x

    def reset_parameters(self):
      # Reset model parameters
        for layers in self.children():
            if hasattr(layers, 'reset_parameters'):
                layers.reset_parameters()
            else:
                for layer in layers:
                    if hasattr(layers, 'reset_parameters'):
                        layers.reset_parameters()