#!/usr/bin/env python

import torch
import argparse
import numpy as np
import torch.nn as nn
from DEAPDataset import DEAPDataset
from models.TestModel import TestModel
from models.MLP import MLP
from models.GCNMLP import GCNMLP
from models.GatedGraphConvMLP import GatedGraphConvMLP
from models.CNN import CNN
from models.GCNCNN import GCNCNN
from models.GatedGraphConvCNN import GatedGraphConvCNN
from models.GRU import GRU
from models.GCNGRU import GCNGRU
from models.GatedGraphConvGRU import GatedGraphConvGRU
from train import main as train_main
from test import main as test_main
from util import get_split_indices

from torchviz import make_dot
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process eeg emotions. http://www.eecs.qmul.ac.uk/mmv/datasets/deap/')
# Common args
parser.add_argument('-rdd', '--raw_data_dir', type=str, default='../data/matlabPREPROCESSED', help='Raw data files directory. (Matlab extension .mat)')
parser.add_argument('-pdd', '--processed_data_dir', type=str, default='./data/processed_data', help='Where to put processed data files from DEAPDataset')
parser.add_argument('-ef', '--eeg_feature', type=str, default='wav', choices=['wav','psd','wav_entropy'], help='Signal preprocessing method for EEG signals')
parser.add_argument('-t', '--target', type=str, default='participant_id', choices=['participant_id','video_id','emotion_labels'], help='Label value for each video. Either participant id or video id')
parser.add_argument('-p', '--participant', type=int, default=None, choices=list(range(1,33)), help='Participant to use')
parser.add_argument('-te', '--target_emotion', type=int, default=0, choices=list(range(0,4)), help='Valence/Arousal/Dominance/Liking')
parser.add_argument('-rgc','--remove_global_connections', default=False, action='store_true',help='Remove global connections from the graph adjacency matrix')
parser.add_argument('-rea','--remove_edge_attributes', default=False, action='store_true',help='Set all values in adjacency matrix to 1')
parser.add_argument('-rbsnr','--remove_baseline_signal_noise_removal', default=False, action='store_true',help='Dont use baseline noise reduction. Simply chop first 3 seconds')
parser.add_argument('-ntt', '--number_test_targets', type=int, default=10, help='Number of test samples. Depends on chosen target')
parser.add_argument('-nvt', '--number_validation_targets', type=int, default=5, help='Number of validation samples. Depends on chosen target')
parser.add_argument('-dev', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda','cpu'], help='Device')
parser.add_argument('-dt', '--dont_train', default=False, action='store_true', help='Load checkpoint and test')
parser.add_argument('-bs', '--batch_size', type=int, default=16)
parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Print logs to console')
parser.add_argument('-kfvo', '--kfold_validation_offset', type=int, default=0)

# Train args
parser.add_argument('-m', '--model', type=str, default='MLP', choices=['MLP','GCNMLP','GatedGraphConvMLP','CNN','GCNCNN','GatedGraphConvCNN','GRU','GCNGRU', 'GatedGraphConvGRU'], help='Which model architecture to train')
parser.add_argument('-hc', '--hidden_channels', type=int, default=32, help='Number of hidden channels in GNN and FCN')
parser.add_argument('-opt', '--optimizer', type=str, default='Adam', choices=['Adam','Adagrad','SGD'])
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
parser.add_argument('-dr', '--dropout_rate', type=float, default=0.4)
parser.add_argument('-lrd', '--learning_rate_decay', type=float, default=0)
parser.add_argument('-wd', '--weight_decay', type=float, default=0)
parser.add_argument('-esp', '--early_stopping_patience', type=int, default=50)
parser.add_argument('-st', '--shuffle_train', default=False, action='store_true', help='Shuffle train dataloader')
parser.add_argument('-me', '--max_epoch', type=int, default=10000)

# Test args
parser.add_argument('-tmd', '--test_model_dict', type=str, default='best_params_tmp', help='Model to test')
parser.add_argument('-wtr', '--write_test_results', default=False, action='store_true', help='Log results to csv')
parser.add_argument('-trd', '--test_results_dir', type=str, default='test1')

args = parser.parse_args()
dataset = DEAPDataset(args)

# dataset = dataset.shuffle()

train_mask,test_mask = get_split_indices(args.target,args.number_test_targets, len(dataset), args.kfold_validation_offset )

train_dataset = dataset[train_mask]
test_dataset = dataset[test_mask]

print(f'Train/Val dataset: {train_dataset} | Test dataset: {test_dataset}',)

print(f'Device: {args.device}')

in_channels = dataset[0].x.shape[1]
n_graphs = dataset[0].x.shape[0]//32
if args.target == 'emotion_labels':
    n_classes = 2
else:
    n_classes = np.unique(np.array([d.y for d in dataset])).shape[0]

criterion = nn.CrossEntropyLoss()
print(f'Number of classes: {n_classes}')


if args.model == 'MLP':
    model = MLP(in_channels,n_graphs,args.hidden_channels, n_classes, args.dropout_rate)
elif args.model == 'GCNMLP':
    model = GCNMLP(in_channels,n_graphs,args.hidden_channels, n_classes, args.dropout_rate)
elif args.model == 'GatedGraphConvMLP':
    model = GatedGraphConvMLP(in_channels,n_graphs,args.hidden_channels, n_classes, args.dropout_rate)
elif args.model == 'CNN':
    model = CNN(in_channels,n_graphs,args.hidden_channels, n_classes, args.dropout_rate)
elif args.model == 'GCNCNN':
    model = GCNCNN(in_channels,n_graphs,args.hidden_channels, n_classes, args.dropout_rate)
elif args.model == 'GatedGraphConvCNN':
    model = GatedGraphConvCNN(in_channels,n_graphs,args.hidden_channels, n_classes, args.dropout_rate)
elif args.model == 'GRU':
    model = GRU(in_channels,n_graphs,args.hidden_channels, n_classes, args.dropout_rate)
elif args.model == 'GCNGRU':
    model = GCNGRU(in_channels,n_graphs,args.hidden_channels, n_classes, args.dropout_rate)
elif args.model == 'GatedGraphConvGRU':
    model = GatedGraphConvGRU(in_channels,n_graphs,args.hidden_channels, n_classes, args.dropout_rate)

print(model)

make_dot(model(dataset[0]), params=dict(model.named_parameters())).render(f"dot/{args.model}", format="png")


pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'Model parameter count: {pytorch_total_params}')

# if not args.dont_train:
#     train_main(model,train_dataset,criterion,args)

# test_main(model,test_dataset,criterion,args)

