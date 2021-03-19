import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

from models.STGCN.TemporalGatedConv  import TemporalGatedConv
# from models.STGCN.SpatialGraphConv  import SpatialGraphConv
from torch_geometric.nn import GraphConv

class SpatioTemporalBlock(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,kernel_size,stride=1):
        super(SpatioTemporalBlock, self).__init__()

        # Temporal gated convs
        self.tgc1 = TemporalGatedConv(in_channels,out_channels,kernel_size)

        # Spatial graph convs
        self.sgc = GraphConv(out_channels,hidden_channels)

        # Temporal gated convs
        self.tgc2 = TemporalGatedConv(hidden_channels,out_channels,kernel_size)

    def forward(self, x,edge_index,edge_attr,batch):
        x = self.tgc1(x)
        bs = x.shape[0]

        x = self.sgc(x,edge_index,edge_attr)

        x = self.tgc2(x)
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