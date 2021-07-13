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

parser = argparse.ArgumentParser(description='Process eeg emotions. http://www.eecs.qmul.ac.uk/mmv/datasets/deap/')
# Common args
parser.add_argument('-rdd', '--raw_data_dir', type=str, default='../../data/matlabPREPROCESSED', help='Raw data files directory. (Matlab extension .mat)')
parser.add_argument('-pdd', '--processed_data_dir', type=str, default='../data/processed_data', help='Where to put processed data files from DEAPDataset')
parser.add_argument('-ef', '--eeg_feature', type=str, default='wav', choices=['wav','psd'], help='Feature extraction method for EEG signals')
parser.add_argument('-t', '--target', type=str, default='participant_id', choices=['participant_id','video_id','emotions_combined','valence','arousal','dominance','liking'], help='Target for samples')
parser.add_argument('-p', '--participant', type=int, default=None, choices=list(range(1,33)), help='If != None it specifies an individual participant to use as raw data')
# parser.add_argument('-te', '--target_emotion', type=int, default=0, choices=list(range(0,4)), help='Valence/Arousal/Dominance/Liking')
parser.add_argument('-rgc','--remove_global_connections', default=False, action='store_true',help='Remove global connections from the graph adjacency matrix')
parser.add_argument('-ts', '--train_split', type=int, default=10, help='Percentage of the total data that will be used for training. If set to -1 just one example per class will be picked, this is only valid for participant id and video id classification')
parser.add_argument('-vs', '--val_split', type=int, default=10, help='Percentage of the total data that will be used for validation. Remaining data will be used for testing')
# parser.add_argument('-rea','--remove_edge_attributes', default=False, action='store_true',help='Set all values in adjacency matrix to 1')
# parser.add_argument('-rbsnr','--remove_baseline_signal_noise_removal', default=True, action='store_false',help='Dont use baseline noise reduction. Simply chop first 3 seconds')
# parser.add_argument('-ntt', '--number_test_targets', type=int, default=10, help='Number of test samples. Depends on chosen target')
# parser.add_argument('-nvt', '--number_validation_targets', type=int, default=5, help='Number of validation samples. Depends on chosen target')
parser.add_argument('-dev', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda','cpu'], help='Device')
parser.add_argument('-dt', '--dont_train', default=False, action='store_true', help='Load checkpoint and test')
parser.add_argument('-bs', '--batch_size', type=int, default=16)
parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Print training logs to console')
parser.add_argument('-kfvo', '--kfold_validation_offset', type=int, default=0)
parser.add_argument('-ws', '--window_size', type=float, default=1.0, help='Size of the windows taken from each video (seconds). Should be divisible by 60')


# Train args
parser.add_argument('-m', '--model', type=str, default='MLP', choices=['MLP','GCNMLP','GatedGraphConvMLP','CNN','GCNCNN','GatedGraphConvCNN','GRU','GCNGRU', 'GatedGraphConvGRU'], help='Which model architecture to train')
parser.add_argument('-hc', '--hidden_channels', type=int, default=64, help='Number of hidden channels')
parser.add_argument('-opt', '--optimizer', type=str, default='Adam', choices=['Adam','Adagrad','SGD'])
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-dr', '--dropout_rate', type=float, default=0.4)
parser.add_argument('-lrd', '--learning_rate_decay', type=float, default=0)
parser.add_argument('-wd', '--weight_decay', type=float, default=0)
parser.add_argument('-esp', '--early_stopping_patience', type=int, default=50)
parser.add_argument('-st', '--shuffle_train', default=False, action='store_true', help='Shuffle train dataloader')
parser.add_argument('-me', '--max_epoch', type=int, default=10000)

# Test args
parser.add_argument('-tmd', '--test_model_dict', type=str, default='../checkpoints/checkpoint1', help='Model to test')
parser.add_argument('-wtr', '--write_test_results', default=False, action='store_true', help='Log results to csv')
parser.add_argument('-trd', '--test_results_dir', type=str, default='test1')

args = parser.parse_args()


dataset = DEAPDataset(args)

train_mask, val_mask, test_mask = get_split_indices(args.target, args.train_split, args.val_split , len(dataset),args.window_size, args.kfold_validation_offset)

train_dataset = dataset[train_mask]
val_dataset = dataset[val_mask]
test_dataset = dataset[test_mask]

print(f'Device: {args.device}')

in_channels = dataset[0].x.shape[1]
if args.target in ['valence','arousal','dominance','liking']:
    n_classes = 2
else:
    n_classes_train = np.unique(np.array([d.y for d in train_dataset])).shape[0]
    n_classes = np.unique(np.array([d.y for d in dataset])).shape[0]

criterion = nn.CrossEntropyLoss()
assert n_classes_train == n_classes
print(f'Using {args.eeg_feature} as eeg feature')
print(f'Number of classes in train dataset: {n_classes}')


if args.model == 'MLP':
    model = MLP(in_channels,args.hidden_channels, n_classes, args.dropout_rate).to(args.device) 
elif args.model == 'GCNMLP':
    model = GCNMLP(in_channels,args.hidden_channels, n_classes, args.dropout_rate).to(args.device) 
elif args.model == 'GatedGraphConvMLP':
    model = GatedGraphConvMLP(in_channels,n_graphs,args.hidden_channels, n_classes, args.dropout_rate).to(args.device) 
elif args.model == 'CNN':
    model = CNN(in_channels,args.hidden_channels, n_classes, args.dropout_rate).to(args.device) 
elif args.model == 'GCNCNN':
    model = GCNCNN(in_channels,n_graphs,args.hidden_channels, n_classes, args.dropout_rate).to(args.device) 
elif args.model == 'GatedGraphConvCNN':
    model = GatedGraphConvCNN(in_channels,args.hidden_channels, n_classes, args.dropout_rate).to(args.device) 
elif args.model == 'GRU':
    model = GRU(in_channels,n_graphs,args.hidden_channels, n_classes, args.dropout_rate).to(args.device) 
elif args.model == 'GCNGRU':
    model = GCNGRU(in_channels,n_graphs,args.hidden_channels, n_classes, args.dropout_rate).to(args.device) 
elif args.model == 'GatedGraphConvGRU':
    model = GatedGraphConvGRU(in_channels,n_graphs,args.hidden_channels, n_classes, args.dropout_rate).to(args.device) 

print(model)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'Model parameter count: {pytorch_total_params}')
# print(f'K fold offset: {args.kfold_validation_offset}')
print(f'Train dataset: {train_dataset} | Validation dataset: {val_dataset} | Test dataset: {test_dataset}',)

if not args.dont_train:
    train_main(model,train_dataset,val_dataset,criterion,args)

test_main(model,test_dataset,criterion,args)

