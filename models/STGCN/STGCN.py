import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
import matplotlib.pyplot as plt

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

        # Spatio-temporal blocks
        self.stb1 = SpatioTemporalBlock(in_channels=self.window_size,hidden_channels=16,out_channels=64,kernel_size=15)
        self.stb2 = SpatioTemporalBlock(in_channels=64,hidden_channels=8,out_channels=64,kernel_size=16)
        # self.stb3 = SpatioTemporalBlock(in_channels=128,hidden_channels=32,out_channels=128,kernel_size=8)
        # self.stb4 = SpatioTemporalBlock(in_channels=128,hidden_channels=32,out_channels=128,kernel_size=8)
        # self.stb5 = SpatioTemporalBlock(in_channels=128,hidden_channels=32,out_channels=128,kernel_size=2)
        self.conv = nn.Conv1d(64 ,1, 2)
        self.fc = nn.Linear(32,1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, args):

        x = batch.x
        edge_index = batch.edge_index[:,:194]
        edge_attr = batch.edge_attr[:194]
        batch = batch.batch
        N = len(torch.unique(batch))

        r"""
                N is batch size,
                M is the number of time windows
                n is the electrode count (32)
                c is the number of signal datapoints per window
        """
        # Divide into time windows 
        # Shape after reshape and 1 second (128Hz) windows
        # torch.Size([N, 60, 32, 128])
        x = rearrange(x,'(N n) (M c) -> N M n c',N=N, c = self.window_size)

        x = self.stb1(x,edge_index,edge_attr,batch)
        x = x.relu()
        if args.visualize_convs:
            plt.matshow(x[0,0,:,:].cpu().detach().numpy())
            plt.show()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.stb2(x,edge_index,edge_attr,batch)
        x = x.relu()
        if args.visualize_convs:
            plt.matshow(x[0,0,:,:].cpu().detach().numpy())
            plt.show()
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x.shape)
        # x = self.stb3(x,edge_index,edge_attr,batch)
        # x = x.relu()
        # if args.visualize_convs:
        #     plt.matshow(x[0,0,:,:].cpu().detach().numpy())
        #     plt.show()
        # x = self.stb4(x,edge_index,edge_attr,batch)
        # x = x.relu()
        # if args.visualize_convs:
        #     plt.matshow(x[0,0,:,:].cpu().detach().numpy())
        #     plt.show()
        # x = self.stb5(x,edge_index,edge_attr,batch)
        # x = x.relu()
        # if args.visualize_convs:
        #     plt.matshow(x[0,0,:,:].cpu().detach().numpy())
        #     plt.show()
        # x = F.dropout(x, p=0.3, training=self.training)

        # print(x.shape)
        # exit()
        
        x = rearrange(x,'N M n c -> (N n) c M')

        x = self.conv(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = rearrange(x,'(N n) c M ->N (n c M)',N=N)
        # x = gap(x,batch)
        # x = torch.clamp(x,0,10)
        # x = x.relu()

        x = self.fc(x)
        x = torch.sigmoid(x)
       
        return x

        
