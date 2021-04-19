import os
import scipy
import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange
from torch_geometric.data import InMemoryDataset, Data
from Electrodes import Electrodes
from sklearn.preprocessing import MinMaxScaler

def create_data_for_participant(participant_data,  window_size, target_length=1, sequence_distance=5, n_videos=40):
    """
    For a certain participant, create input and target features
    :param participant_data: Original participant_data. Shape (40, 32, 7680)
    :param look_back: Number of windows included as input features
    :param window_size: Number of data points per window
    :param overlap_sequences: Wether to add 1 or window_size to get the next sequence.
    :return: 
        inputs: Input features with shape (32*look_back, window_size)
        targets: Targets with shape (32, window_size)
    """
    inputs = []
    targets = []
    for video in participant_data[:n_videos]:
        idx = 0
        while idx + window_size + target_length <= video.shape[1]:
            inpt = video[:,idx:idx+window_size]
            inpt = rearrange(inpt,'a (b c) -> (b a) c',c=window_size)
            tget = video[:,idx+window_size:idx+window_size+target_length]
            inputs.append(inpt)
            targets.append(tget)
            idx += sequence_distance
    return inputs,targets

class DEAPDatasetSequence(InMemoryDataset):
 
  def __init__(self, root, raw_dir,processed_dir, transform=None, pre_transform=None,include_edge_attr = True, undirected_graphs = True, add_global_connections=True, participant_n = 1, n_videos=5, window_size=128*5,target_length=16):
      self._raw_dir = raw_dir
      self._processed_dir = processed_dir
      self.participant_from = participant_n
      self.participant_to = participant_n
      self.n_videos = n_videos
      self.target_length = target_length
    #   self.look_back = look_back
      self.window_size = window_size
      # Whether or not to include edge_attr in the dataset
      self.include_edge_attr = include_edge_attr
      # If true there will be 1024 links as opposed to 528
      self.undirected_graphs = undirected_graphs
      # Instantiate class to handle electrode positions
      print('Using global connections' if add_global_connections else 'Not using global connections')
      self.electrodes = Electrodes(add_global_connections, expand_3d = False)
      super(DEAPDatasetSequence, self).__init__(root, transform, pre_transform)
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
      file_name = f'{self.n_videos}'
      return [f'deap_processed_graph.{file_name}.dataset']

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
        # e_edge_index = edge_index.clone().detach()
        # e_edge_attr = edge_attr.clone().detach()
        # for i in range(self.look_back-1):
        #     a = edge_index + e_edge_index.max() + 1
        #     e_edge_index = torch.cat([e_edge_index,a],dim=1)
        #     e_edge_attr = torch.cat([e_edge_attr,edge_attr],dim=0)

        # List of graphs that will be written to file
        data_list = []
        pbar = tqdm(range(self.participant_from,self.participant_to+1))
        for participant_id in pbar:
            raw_name = [e for e in self.raw_file_names if str(participant_id).zfill(2) in e][0]
            pbar.set_description(raw_name)
            # Load raw file as np array
            participant_data = scipy.io.loadmat(f'{self.raw_dir}/{raw_name}')
            signal_data = torch.FloatTensor(participant_data['data'][:,:32,128*3:])
            print(signal_data.shape)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            # Maybe this should be fit just to testing data
            signal_data_normalized = scaler.fit_transform(signal_data.reshape(-1, 1))
            signal_data = rearrange(signal_data_normalized,'(a b c) d -> a b (c d)',b=32,a=40)
            signal_data = torch.FloatTensor(signal_data)
            print(signal_data.shape)

            inputs, targets = create_data_for_participant(signal_data,n_videos=self.n_videos,window_size=self.window_size,target_length =self.target_length )
            for i, t in zip(inputs,targets):
                data = Data(x=i,edge_attr=edge_attr,edge_index=edge_index, y=t) if self.include_edge_attr else Data(x=i, edge_index=edge_index, y=t)
                data_list.append(data)   
               
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])