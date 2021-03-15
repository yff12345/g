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

        # Spatio-temporal block 1
        self.stb1 = SpatioTemporalBlock(in_channels=self.window_size,hidden_channels=64,out_channels=256,kernel_size=16)
        self.stb2 = SpatioTemporalBlock(in_channels=256,hidden_channels=64,out_channels=128,kernel_size=15)
        # self.stb3 = SpatioTemporalBlock(in_channels=64,hidden_channels=32,out_channels=64,kernel_size=2)
        # self.stb4 = SpatioTemporalBlock(in_channels=64,hidden_channels=32,out_channels=64,kernel_size=8)
        # self.stb5 = SpatioTemporalBlock(in_channels=64,hidden_channels=32,out_channels=64,kernel_size=2)
        self.conv = nn.Conv1d(128 ,1, 2)
        # self.fc = nn.Linear(128,1)

        self.temp = 0

        self.sigmoid = nn.Sigmoid()

        self.reset_model(False)

    def reset_model(self,reset_params = True):
      # Reset model parameters
      if reset_params:
        for layers in self.children():
          if hasattr(layers, 'reset_parameters'):
            layers.reset_parameters()
          else:
            for layer in layers:
              if hasattr(layers, 'reset_parameters'):
                layers.reset_parameters()
      self.best_val_mse = float('inf')
      self.train_losses = []
      self.eval_losses = []
      self.eval_patience_count = 0
      self.eval_patience_reached = False

    def forward(self, batch):

        r"""Spatial temporal graph convolutional networks.
        Shape:
            - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
            - Output: :math:`(N, num_class)` where
                :math:`N` is a batch size,
                :math:`T_{in}` is a length of input sequence,
                :math:`V_{in}` is the number of graph nodes,
                :math:`M_{in}` is the number of instance in a frame.
        """
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch = batch.batch
        bs = len(torch.unique(batch))

        # Divide into time windows 
        # Y ∈ R M×n×Ci
        # Number of windows x number of electrodes x number of channels
        # bs: batch size ec: electrode count  wn: window count cs: channel size
        # il: input length
        x = rearrange(x,'(bs ec) (wc cs) -> (bs ec) cs wc',bs=bs, cs = self.window_size)

 
        x = self.stb1(x,edge_index,edge_attr,batch)
        # print(x)
        x = self.stb2(x,edge_index,edge_attr,batch)
        # print(x)
        # print(x.shape)
        # x = self.stb3(x,edge_index,edge_attr,batch)
        # x = self.stb4(x,edge_index,edge_attr,batch)
        # x = self.stb5(x,edge_index,edge_attr,batch)
        
        x = F.dropout(x, p=0.3, training=self.training)

        # print(x)
        x = gap(x,batch)
        # print(x)
        x = x.relu()
        
        # print(x.shape)
        x = self.conv(x)
        # print(x)
        
        # print(x.shape)
        x = rearrange(x,'bs cs wc -> bs (cs wc)',bs=bs)
        # x = self.fc(x)

        x = self.sigmoid(x)
        x = torch.multiply(x,10)
        # print(x)

        # self.temp += 1
        # if self.temp == 20:
        #   exit()
       
        return x
        
