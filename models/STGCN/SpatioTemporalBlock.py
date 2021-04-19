import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

from models.STGCN.TemporalGatedConv  import TemporalGatedConv
# from models.STGCN.SpatialGraphConv  import SpatialGraphConv
from torch_geometric.nn import GCNConv

class SpatioTemporalBlock(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,kernel_size,stride=1):
        super(SpatioTemporalBlock, self).__init__()

        # Temporal gated convs
        self.tgc1 = TemporalGatedConv(in_channels,out_channels,kernel_size)

        # Spatial graph convs
        self.sgc = GCNConv(out_channels,hidden_channels)

        # Temporal gated convs
        self.tgc2 = TemporalGatedConv(hidden_channels,out_channels,kernel_size)

    def forward(self, x,edge_index,edge_attr,batch):
        x = self.tgc1(x)
        N,M = x.shape[0],x.shape[1]
        x = rearrange(x,' N M n c -> (N M n) c')
        # print(N*M)
        # print(x.shape)
        # print(edge_attr.shape)
        # Expand edge index and edge attr
        e_edge_index = torch.tensor(edge_index)
        e_edge_attr = torch.tensor(edge_attr)
        for i in range((N*M)-1):
            a = edge_index + e_edge_index.max() + 1
            b = edge_attr + e_edge_attr.max() + 1 
            e_edge_index = torch.cat([e_edge_index,a],dim=1)
            e_edge_attr = torch.cat([e_edge_attr,b],dim=0)
            # THIS IS WRONG. EDGE ATTR SHOULD REPEAT AND NOT INCREMENT.
            raise 'err'
        
        # print(e_edge_attr)


        x = self.sgc(x,e_edge_index,e_edge_attr)
        x = rearrange(x,'(N M n) c -> N M n c',N=N,M=M)

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