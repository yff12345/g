import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

from models.STGCN.TemporalGatedConv  import TemporalGatedConv
# from models.STGCN.SpatialGraphConv  import SpatialGraphConv
from torch_geometric.nn import GCNConv

class SpatioTemporalBlock(torch.nn.Module):
    def __init__(self,in_channels):
        super(SpatioTemporalBlock, self).__init__()

        # Temporal gated convs
        self.tgc1 = TemporalGatedConv(in_channels,64,8)
        # Temporal gated convs
        self.tgc2 = TemporalGatedConv(16,64,8)

        # Spatial graph convs
        self.sgc = GCNConv(64,16)

    def forward(self, x,edge_index,edge_attr,batch):
        x = self.tgc1(x)
        # X ∈ RM×n×Ci
        x = rearrange(x,'ec cs wc -> wc ec cs')
        x = F.relu(self.sgc(x,edge_index,edge_attr))
        x = rearrange(x,'wc ec cs -> ec cs wc')
        x = self.tgc2(x)
        return x