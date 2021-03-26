import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

from torch_geometric.nn import global_add_pool as gap

from models.STGCN.SpatioTemporalBlock import SpatioTemporalBlock


class STGCN(torch.nn.Module):
    r"""Spatial temporal graph convolutional networks.
      Args:
          window_size (int): Number of Hz per window
      """

    def __init__(self, window_size):
        super(STGCN, self).__init__()

        self.window_size = window_size
        # Early stopping
        self.best_val_loss = float('inf')
        self.eval_patience_count = 0
        self.eval_patience_reached = False
        # Record metrics
        self.train_losses = []
        self.eval_losses = []
        self.best_epoch = -1

        # Spatio-temporal block 1
        self.stb1 = SpatioTemporalBlock(in_channels=self.window_size,hidden_channels=32,out_channels=64,kernel_size=8)
        self.stb2 = SpatioTemporalBlock(in_channels=64,hidden_channels=16,out_channels=64,kernel_size=8)
        self.stb3 = SpatioTemporalBlock(in_channels=64,hidden_channels=16,out_channels=64,kernel_size=8)
        self.stb4 = SpatioTemporalBlock(in_channels=64,hidden_channels=16,out_channels=64,kernel_size=8)
        self.stb5 = SpatioTemporalBlock(in_channels=64,hidden_channels=16,out_channels=64,kernel_size=2)
        self.conv = nn.Conv1d(64 ,1, 2)
        # self.fc = nn.Linear(64,1)

        self.sigmoid = nn.Sigmoid()

        self.reset_model(False)

    def forward(self, batch):

        x = batch.x
        edge_index = batch.edge_index[:,:194]
        edge_attr = batch.edge_attr[:194]
        batch = batch.batch
        bs = len(torch.unique(batch))

        r"""
                N is batch size,
                M is the number of time windows
                n is the electrode count (32)
                c is the number of signal datapoints per window
        """
        # Divide into time windows 
        # Shape after reshape and 1 second (128Hz) windows
        # torch.Size([N, 60, 32, 128])
        x = rearrange(x,'(N n) (M c) -> N M n c',N=bs, c = self.window_size)

        x = self.stb1(x,edge_index,edge_attr,batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.stb2(x,edge_index,edge_attr,batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.stb3(x,edge_index,edge_attr,batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.stb4(x,edge_index,edge_attr,batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.stb5(x,edge_index,edge_attr,batch)

        x = rearrange(x,'N M n c -> (N n) c M')
        x = x.relu()
        x = self.conv(x)
        x = rearrange(x,'n c M -> n (c M)')
        x = gap(x,batch)
        # x = x.relu()

        # x = self.fc(x)
        x = torch.sigmoid(x)
       
        return x

        
