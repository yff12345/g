import os
import torch
import numpy as np
import scipy.io
import skimage
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
from Electrodes import Electrodes
from einops import repeat
from signal_processing import process_window_wavelet, process_window_psd

class DEAPDataset(InMemoryDataset):
    def __init__(self, args, participant_from=1, participant_to=32, n_videos=40):
        self._raw_dir = args.raw_data_dir
        self._processed_dir = args.processed_data_dir
        self.participant_from = args.participant if args.participant is not None else participant_from
        self.participant_to = args.participant if args.participant is not None else participant_to
        self.feature = args.eeg_feature
        self.target = args.target
        self.n_videos = n_videos
        self.include_edge_attr = True
        self.window_size = args.window_size
        # Define electrode class to get graph adjacency matrix
        self.electrodes = Electrodes(not args.remove_global_connections)
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
      return [f'deap_processed_graph.{self.participant_from}-{self.participant_to}_{self.feature}_{self.target}_{self.window_size}s.dataset']

    def process(self):
        # Number of nodes per graph
        n_nodes = len(self.electrodes.positions_3d)

        # Define links for all graphs
        source_nodes, target_nodes = np.tril_indices(n_nodes,n_nodes)
        edge_attr = self.electrodes.adjacency_matrix[source_nodes,target_nodes]
        # Remove zero weight links
        mask = np.ma.masked_not_equal(edge_attr, 0).mask
        edge_attr,source_nodes,target_nodes = edge_attr[mask], source_nodes[mask], target_nodes[mask]
        edge_attr, edge_index = torch.FloatTensor(edge_attr), torch.tensor([source_nodes,target_nodes], dtype=torch.long)

        data_list = []
        pbar = tqdm(range(self.participant_from,self.participant_to+1))
        for participant_id in pbar:
            # Read raw data
            raw_name = [e for e in self.raw_file_names if str(participant_id).zfill(2) in e][0]
            pbar.set_description(raw_name)
            # Load raw file as np array
            participant_data = scipy.io.loadmat(f'{self.raw_dir}/{raw_name}')
            # Keep just EEG signals and remove 3 second baseline
            signal_data = torch.FloatTensor(participant_data['data'][:,:32,128*3:])
            processed = []
            for video_index, video in enumerate(signal_data[:self.n_videos,:,:]):
                ## Define targets for features
                if self.target == 'participant_id':
                    target = participant_id -1
                elif self.target == 'video_id':
                    target = video_index
                elif self.target in ['valence','arousal','dominance','liking']:
                    emotions = ['valence','arousal','dominance','liking']
                    target = participant_data['labels'][video_index][emotions.index(self.target)]
                else: # emotions_combined
                    target = participant_data['labels'][video_index]
                    target_binary = np.array(target>5,dtype=int)
                    target_binary = ''.join([str(x) for x in target_binary])
                    target = int(target_binary, 2)

                ## Split videos into windows and extract features
                assert 60 % self.window_size == 0
                # Non-overlapping windows
                video_windows = skimage.util.view_as_windows(video.numpy(), (32,int(self.window_size*128)), step=int(self.window_size*128)).squeeze()
                for window in video_windows:
                    node_features =  process_window_wavelet(window) if self.feature == 'wav' else process_window_psd(window)
                    data = Data(x=node_features,edge_attr=edge_attr,edge_index=edge_index, y=torch.LongTensor([target])) if self.include_edge_attr else Data(x=torch.FloatTensor(node_features), edge_index=edge_index, y=torch.LongTensor([target]))
                    data_list.append(data) 
               
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])