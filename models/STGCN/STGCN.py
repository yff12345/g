import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

from models.STGCN.SpatioTemporalBlock import SpatioTemporalBlock


class STGCN(torch.nn.Module):
    def __init__(self):
        super(STGCN, self).__init__()

        # Spatio-temporal block 1
        self.stb1 = SpatioTemporalBlock(128)
        self.stb2 = SpatioTemporalBlock(64)
        self.conv = nn.Conv1d(64 ,64, 32)
        self.fc = nn.Linear(2048,1)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch = batch.batch
        bs = len(torch.unique(batch))

        # Divide into time windows 
        # Y ∈ R M×n×Ci
        # Number of windows x number of electrodes x number of channels
        # bs: batch size ec: electrode count  wn: window count cs: channel size
        x = rearrange(x,'(bs ec) (wc cs) -> (bs ec) cs wc',bs=bs, cs = 128)

        x = self.stb1(x,edge_index,edge_attr,batch)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.stb2(x,edge_index,edge_attr,batch)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv(x)
        x = rearrange(x,'(bs ec) cs wc -> bs (ec cs wc)',bs=bs)
        x = F.relu(self.fc(x))
        return x
        
