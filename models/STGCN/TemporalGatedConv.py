import torch
import torch.nn.functional as F
import torch.nn as nn

class TemporalGatedConv(torch.nn.Module):
    def __init__(self , in_channels, out_channels, kernel_size):
        super(TemporalGatedConv, self).__init__()

        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels , 2*out_channels, kernel_size)

    def reset_parameters(self):
      # Reset model parameters
        for layers in self.children():
            if hasattr(layers, 'reset_parameters'):
                layers.reset_parameters()
            else:
                for layer in layers:
                    if hasattr(layers, 'reset_parameters'):
                        layers.reset_parameters()

    def forward(self, x):
        # (N,C_in,L_in)

        x = self.conv(x)

        # (N,C_out,L_out)
        # P = x[:,:self.out_channels,:]
        # Q = x[:,self.out_channels:,:]
        
        return F.glu(x,dim=1)