import os
import torch
import numpy as np
import scipy.io
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
from Electrodes import Electrodes
from einops import repeat
from signal_processing import remove_baseline_mean, process_video_wavelet, process_video

class DEAPDataset(InMemoryDataset):
  def __init__(self, args, participant_from=1, participant_to=32, n_videos=40):
      self.args = args
      if args.participant is not None:
        participant_from = args.participant
        participant_to = args.participant
      self._raw_dir = args.raw_data_dir
      self._processed_dir = args.processed_data_dir
      self.feature = args.eeg_feature
      self.target = args.target
      self.participant_from = participant_from
      self.participant_to = participant_to
      self.n_videos = n_videos
      self.include_edge_attr = not args.remove_edge_attributes
      self.electrodes = Electrodes(not args.remove_global_connections, expand_3d = False)
      self.samples_per_video = args.samples_per_video
      transform, pre_transform = None,None
      super(DEAPDataset, self).__init__('./', transform, pre_transform)
      self.data, self.slices = torch.load(self.processed_paths[0])
      
  @property
  def raw_dir(self):
      return self._raw_dir

  @property
  def processed_dir(self):
      return self._processed_dir

  @property
  def raw_file_names(self):
      raw_names = [f for f in os.listdir(self.raw_dir)]
      raw_names.sort()
      return raw_names

  @property
  def processed_file_names(self):
      if not os.path.exists(self.processed_dir):
        os.makedirs(self.processed_dir)
      return [f'deap_processed_graph.{self.participant_from}-{self.participant_to}_{self.feature}_{self.target}_{self.samples_per_video*40}.dataset']

  def process(self):
        # Number of nodes per graph
        n_nodes = len(self.electrodes.positions_3d)
    
        source_nodes, target_nodes = np.tril_indices(n_nodes,n_nodes)
        
        edge_attr = self.electrodes.adjacency_matrix[source_nodes,target_nodes]
        
        # Remove zero weight links
        mask = np.ma.masked_not_equal(edge_attr, 0).mask
        edge_attr,source_nodes,target_nodes = edge_attr[mask], source_nodes[mask], target_nodes[mask]

        edge_attr, edge_index = torch.FloatTensor(edge_attr), torch.tensor([source_nodes,target_nodes], dtype=torch.long)
        
        number_of_graphs = 4
        print(f'-Number of graphs (freq. bands) per sample: {number_of_graphs}')
        
        # Expand edges to match number of frequency bands
        e_edge_index = edge_index.clone()
        e_edge_attr = edge_attr.clone()
        for i in range(number_of_graphs-1):
            a = edge_index + e_edge_index.max() + 1
            e_edge_index = torch.cat([e_edge_index,a],dim=1)
            e_edge_attr = torch.cat([e_edge_attr,edge_attr],dim=0)
        
        # List of graphs that will be written to file
        data_list = []
        pbar = tqdm(range(self.participant_from,self.participant_to+1))
        for participant_id in pbar:
            raw_name = [e for e in self.raw_file_names if str(participant_id).zfill(2) in e][0]
            pbar.set_description(raw_name)
            # Load raw file as np array
            participant_data = scipy.io.loadmat(f'{self.raw_dir}/{raw_name}')
            if self.args.remove_baseline_signal_noise_removal:
                signal_data = torch.FloatTensor(participant_data['data'][:,:32,128*3:])
            else:
                signal_data = torch.FloatTensor(remove_baseline_mean(participant_data['data'][:,:32,:]))
            processed = []
            for video_index, video in enumerate(signal_data[:self.n_videos,:,:]):
                
                ## Define targets for features
                if self.target == 'emotion_labels':
                    target = participant_data['labels'][video_index]
                elif self.target == 'participant_id':
                    target = participant_id-1
                elif self.target == 'video_id':
                    target = video_index
                else:
                    raise 'Invalid target'


                ## Split videos into windows and extract features
                video_length = video.shape[1] 
                window_size = video_length // self.samples_per_video

                for i in range(self.samples_per_video):
                    video_window = video[:,window_size*i:window_size*(i+1)]
                    node_features =  process_video_wavelet(video_window) if self.feature == 'wav' else process_video(video_window,feature=self.feature)
                    data = Data(x=torch.FloatTensor(node_features),edge_attr=e_edge_attr,edge_index=e_edge_index, y=torch.LongTensor([target])) if self.include_edge_attr else Data(x=torch.FloatTensor(node_features), edge_index=e_edge_index, y=torch.LongTensor([target]))
                    data_list.append(data) 
               
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])