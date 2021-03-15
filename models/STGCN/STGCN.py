import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

from models.STGCN.SpatioTemporalBlock import SpatioTemporalBlock


class STGCN(torch.nn.Module):
    def __init__(self):
        super(STGCN, self).__init__()

        # Spatio-temporal block 1
        self.stb1 = SpatioTemporalBlock(in_channels=128,hidden_channels=32,out_channels=64,kernel_size=8)
        self.stb2 = SpatioTemporalBlock(in_channels=64,hidden_channels=16,out_channels=32,kernel_size=8)
        self.conv = nn.Conv1d(32 ,32, 32)
        self.fc = nn.Linear(1024,1)

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
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.stb2(x,edge_index,edge_attr,batch)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # print(x.shape)
        x = self.conv(x)
        x = rearrange(x,'(bs ec) cs wc -> bs (ec cs wc)',bs=bs)
        # print(x.shape)
        
        x = F.relu(self.fc(x))
        return x
        
