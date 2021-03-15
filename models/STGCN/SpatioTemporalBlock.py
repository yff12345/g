import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

from models.STGCN.TemporalGatedConv  import TemporalGatedConv
# from models.STGCN.SpatialGraphConv  import SpatialGraphConv
from torch_geometric.nn import GCNConv

class SpatioTemporalBlock(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,kernel_size):
        super(SpatioTemporalBlock, self).__init__()

        # Temporal gated convs
        self.tgc1 = TemporalGatedConv(in_channels,hidden_channels,kernel_size)

        # Spatial graph convs
        self.sgc = GCNConv(hidden_channels,hidden_channels//4)

        # Temporal gated convs
        self.tgc2 = TemporalGatedConv(hidden_channels//4,out_channels,kernel_size)



    def reset_parameters(self):
      # Reset model parameters
        for layers in self.children():
            if hasattr(layers, 'reset_parameters'):
                layers.reset_parameters()
            else:
                for layer in layers:
                    if hasattr(layers, 'reset_parameters'):
                        layers.reset_parameters()

    def forward(self, x,edge_index,edge_attr,batch):
        x = self.tgc1(x)
        # X ∈ RM×n×Ci
        x = rearrange(x,'ec cs wc -> wc ec cs')
        x = F.relu(self.sgc(x,edge_index,edge_attr))
        x = rearrange(x,'wc ec cs -> ec cs wc')
        x = self.tgc2(x)
        return x