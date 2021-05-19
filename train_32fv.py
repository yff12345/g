#!/usr/bin/env python

import os
import torch
import skimage
import pywt
import scipy.io
import scipy.signal
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats
from einops import reduce, rearrange, repeat
from npeet import entropy_estimators as ee
from torch.optim.lr_scheduler import StepLR
from scipy.fft import rfft, rfftfreq, ifft
from einops import rearrange
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from Electrodes import Electrodes
from tqdm import tqdm

class DEAPDatasetEEGFeatures(InMemoryDataset):
  def __init__(self, root, raw_dir, processed_dir, feature='de', transform=None, pre_transform=None,include_edge_attr = True, undirected_graphs = True, add_global_connections=True, participant_from=1, participant_to=32, n_videos=40):
      self._raw_dir = raw_dir
      self._processed_dir = processed_dir
      self.participant_from = participant_from
      self.participant_to = participant_to
      self.n_videos = n_videos
      self.feature = feature
      # Whether or not to include edge_attr in the dataset
      self.include_edge_attr = include_edge_attr
      # If true there will be 1024 links as opposed to 528
      self.undirected_graphs = undirected_graphs
      # Instantiate class to handle electrode positions
      print('Using global connections' if add_global_connections else 'Not using global connections')
      self.electrodes = Electrodes(add_global_connections, expand_3d = False)
      super(DEAPDatasetEEGFeatures, self).__init__(root, transform, pre_transform)
      self.data, self.slices = torch.load(self.processed_paths[0])
      
  @property
  def raw_dir(self):
      return f'{self.root}/{self._raw_dir}'

  @property
  def processed_dir(self):
      return f'{self.root}/{self._processed_dir}'

  @property
  def raw_file_names(self):
      raw_names = [f for f in os.listdir(self.raw_dir)]
      raw_names.sort()
      return raw_names

  @property
  def processed_file_names(self):
      if not os.path.exists(self.processed_dir):
        os.makedirs(self.processed_dir)
      file_name = f'{self.participant_from}-{self.participant_to}' if self.participant_from is not self.participant_to else f'{self.participant_from}'
      return [f'deap_processed_graph.{file_name}_{self.feature}.dataset']

  def process(self):
        # Number of nodes per graph
        n_nodes = len(self.electrodes.positions_3d)
        

        if self.undirected_graphs:
            source_nodes, target_nodes = np.repeat(np.arange(0,n_nodes),n_nodes), np.tile(np.arange(0,n_nodes),n_nodes)
        else:
            source_nodes, target_nodes = np.tril_indices(n_nodes,n_nodes)
        
        edge_attr = self.electrodes.adjacency_matrix[source_nodes,target_nodes]
        
        # Remove zero weight links
        mask = np.ma.masked_not_equal(edge_attr, 0).mask
        edge_attr,source_nodes,target_nodes = edge_attr[mask], source_nodes[mask], target_nodes[mask]

        edge_attr, edge_index = torch.FloatTensor(edge_attr), torch.tensor([source_nodes,target_nodes], dtype=torch.long)
        
        # Expand edge_index and edge_attr to match windows
        e_edge_index = edge_index.clone()
        e_edge_attr = edge_attr.clone()
        number_of_graphs = 4
        for i in range(number_of_graphs-1):
            a = edge_index + e_edge_index.max() + 1
            e_edge_index = torch.cat([e_edge_index,a],dim=1)
            e_edge_attr = torch.cat([e_edge_attr,edge_attr],dim=0)

        print(f'Number of graphs per video: {number_of_graphs}')
        # List of graphs that will be written to file
        data_list = []
        pbar = tqdm(range(self.participant_from,self.participant_to+1))
        for participant_id in pbar:
            raw_name = [e for e in self.raw_file_names if str(participant_id).zfill(2) in e][0]
            pbar.set_description(raw_name)
            # Load raw file as np array
            participant_data = scipy.io.loadmat(f'{self.raw_dir}/{raw_name}')
            signal_data = torch.FloatTensor(remove_baseline_mean(participant_data['data'][:,:32,:]))
#             signal_data = torch.FloatTensor()
            processed = []
            for i, video in enumerate(signal_data[:self.n_videos,:,:]):
                if self.feature == 'wav':
                    node_features = process_video_wavelet(video)
                else:
                    node_features = process_video(video, feature=self.feature)
                data = Data(x=torch.FloatTensor(node_features),edge_attr=e_edge_attr,edge_index=e_edge_index, y=torch.FloatTensor([participant_data['labels'][i]])) if self.include_edge_attr else Data(x=torch.FloatTensor(node_features), edge_index=e_edge_index, y=torch.FloatTensor([participant_data['labels'][i]]))
                data_list.append(data) 
               
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# Constants used to define data paths
ROOT_DIR = './'
RAW_DIR = 'data/matlabPREPROCESSED'
PROCESSED_DIR = 'data/graphProcessedData'

dataset = DEAPDatasetEEGFeatures(root= ROOT_DIR, raw_dir= RAW_DIR, processed_dir= PROCESSED_DIR, feature='wav')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f'Device: {device}')

from torch_geometric.nn import GCN2Conv, GCNConv
class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=512, cnn_hidden_dim = 256):
        super(Model, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        self.gconv1 = GCN2Conv(in_channels,1)
        self.gconv2 = GCNConv(in_channels,hidden_channels)
        
#         self.gconv3 = GCNConv(in_channels,hidden_channels)
        
        # self.rnn = torch.nn.GRU(hidden_channels, rnn_hidden_dim, 2,dropout=0.2, batch_first=True)
        self.cnn1 = torch.nn.Conv1d(4*hidden_channels, hidden_channels, kernel_size=1, stride=1)
        self.cnn2 = torch.nn.Conv1d(hidden_channels, cnn_hidden_dim, kernel_size=1, stride=1)
        
        self.lin1 = torch.nn.Linear(32*cnn_hidden_dim, 1)
#         self.lin2 = torch.nn.Linear(cnn_hidden_dim, 1)

        
    def forward(self, batch):
        bs = len(torch.unique(batch.batch))
        x_, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
#         print(x.shape)
        x = self.gconv1(x_,x_, edge_index, edge_attr)
#         x = x.relu()
        x = self.gconv2(x, edge_index, edge_attr)
        x = x.tanh()
        
#         x = self.gconv3(x, edge_index, edge_attr )
#         print('-0----------------------------------------------------')
#         print(x)
#         x = x.relu()
#         print(x)
#         x = F.dropout(x, p=0.5, training=self.training)
        
        # x = rearrange(x, '(bs a b) c -> (bs b) a c', bs=bs, b=32, c=self.hidden_channels)
        # o, (h_n,c_n) = self.rnn(x)
        x = rearrange(x, '(bs g e) f -> bs (g f) e', bs=bs, e=32)
        x = self.cnn1(x)
        x = x.relu()
        x = self.cnn2(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = rearrange(x, 'bs a b -> bs (a b)', bs=bs)
        
        
        x = self.lin1(x)
#         x = x.relu()
#         x = self.lin2(x)
        x = x.view(-1)
        return x.sigmoid()

def train(loader, target = 0):
    model.train()
    losses = []
    right = 0
    tot = 0
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        y = (batch.y[:,target] > 5).float()
#         y = batch.y[:,target]
        out = model(batch)
        loss = criterion(out,y)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        right += torch.eq(out > .5, y > .5).sum().item()
        tot += y.shape[0]
    return np.array(losses).mean(), right/tot

def test(loader,verbose=False, target = 0):
    model.eval()
    losses = []
    right = 0
    tot = 0
    for batch in loader:
        batch = batch.to(device)
        y = (batch.y[:,target] > 5).float()
#         y = batch.y[:,target]
        out = model(batch)
        if verbose:
            print(out,y)
        loss = criterion(out,y)
        losses.append(loss.item())
        right += torch.eq(out > .5, y > .5).sum().item()
        tot += y.shape[0]
    return np.array(losses).mean(), right/tot


f = open("log.txt", "w")
# Loop over 32 participants

for test_participant in range(32):
    f.write(f"--------------Participant {test_participant}--------------")


    train_dataset = dataset[0:test_participant*40] + dataset[test_participant*40+40:]
    test_dataset = dataset[test_participant*40:test_participant*40+40]

    model = Model(train_dataset[0].x.shape[1]).to(device)  
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameter count: {pytorch_total_params}')

    optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-4, lr_decay=0, weight_decay=5e-2)

    criterion = nn.BCELoss()

    best_val_loss = np.inf
    esp = 0
    MAX_ESP = 50

    BS = 8

    target = 0 # Valence-Arousal-Dominance-Liking

    splt_idx = 1160
    train_data, val_data = torch.utils.data.random_split(train_dataset, [splt_idx, len(train_dataset)-splt_idx])

    train_loader = DataLoader(train_data, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BS)
    for epoch in range(1, 10000):    

        # Training and validation
        train_loss, train_acc = train(train_loader, target = target)
        val_loss, val_acc = test(val_loader , target = target)
        print(f'Epoch {epoch};t loss: {train_loss:.5f} ;t acc: {train_acc:.2f} ;v loss: {val_loss:.5f} ;v acc: {val_acc:.2f}')

        # Early stopping and checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            esp = 0
            torch.save(model.state_dict(),'./best_params') 
        else:
            esp += 1
            if esp >= MAX_ESP:
                break
                
        if epoch % 20 == 0:
            test_loader = DataLoader(test_dataset, batch_size=1)
            loss, acc = test(test_loader, True)
            print(f'Test loss: {loss} ; Test acc: {acc}')

    print('Finished training')

    model.load_state_dict(torch.load('./best_params'))
    test_loader = DataLoader(test_dataset, batch_size=1)
    loss, acc = test(train_loader, False,target=target)
    f.write(f'Train loss: {loss} ; Train acc: {acc}')
    loss, acc = test(val_loader, False,target=target)

    f.write(f'Val loss: {loss} ; Val acc: {acc}')
    loss, acc = test(test_loader, True,target=target)
    f.write(f'Test loss: {loss} ; Test acc: {acc}')

f.close()