import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import numpy as np
from torch_geometric.nn import global_mean_pool as gmeanp, global_max_pool as gmaxp, global_add_pool as gaddp
from torch_geometric.nn import GraphConv

from einops import reduce, rearrange

# from layers import GCN, HGPSLPoo
from DEAPDataset import visualize_graph

class GNNLSTM(torch.nn.Module):
  def __init__(self):
    super(GNNLSTM, self).__init__()

    self.gconv1 = GraphConv(in_channels=7680, out_channels=4032, aggr='mean')
    self.gconv2 = GraphConv(in_channels=4032, out_channels=512, aggr='mean')
    # self.gconv3 = GraphConv(in_channels=2016, out_channels=512, aggr='mean')

    self.lstm = nn.LSTM(32, 32, 2,bidirectional=True)

    self.mlp = Sequential(Linear(32768, 128),ReLU(),Linear(128, 1))

    # MODEL CLASS ATTRIBUTES
    self.best_val_loss = float('inf')
    self.best_epoch = 0
    self.train_losses = []
    self.eval_losses = []
    self.eval_patience_count = 0
    self.eval_patience_reached = False
     

  def forward(self, batch, visualize_convolutions = False):
    # SETUP
    x = batch.x
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr
    batch = batch.batch
    bs = len(torch.unique(batch))
    # Information propagation trough graph visualization
    if visualize_convolutions:
      visualize_graph(x[:32])

    # GRAPH CONVOLUTIONS
    x = self.gconv1(x,edge_index,edge_attr)
    x = F.relu(x)
    x = F.dropout(x, p=0.3, training=self.training)
    if visualize_convolutions:
      visualize_graph(x[:32])

    x = self.gconv2(x,edge_index,edge_attr)
    x = F.relu(x)
    x = F.dropout(x, p=0.2, training=self.training)
    if visualize_convolutions:
      visualize_graph(x[:32])
    
    # x = self.gconv3(x,edge_index,edge_attr)
    # x = F.relu(x)
    # # x = F.dropout(x, p=0.2, training=self.training)
    # if visualize_convolutions:
    #   visualize_graph(x[:32])

    # LSTM
    x = rearrange(x,'(bs ec) sl -> sl bs ec',bs=bs)
    output, (c_n,h_n)  = self.lstm(x)

    # MLP
    x = rearrange(output,'sl b i -> b (sl i)')
    x = F.dropout(x, p=0.2, training=self.training)
    # print(x.shape)
    x = self.mlp(x)
    x = x.sigmoid()

    return x