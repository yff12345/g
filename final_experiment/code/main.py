#!/usr/bin/env python

import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from DEAPDataset import DEAPDataset
from models.MLP import MLP
from models.CNN import CNN
from models.GraphConv import GraphConv
from models.GIN import GIN
# from decimals import *

from train import main as train_main
from test import main as test_main
from util import get_split_indices

parser = argparse.ArgumentParser(description='Process eeg emotions. http://www.eecs.qmul.ac.uk/mmv/datasets/deap/')
# Common args
parser.add_argument('-rdd', '--raw_data_dir', type=str, default='../../data/matlabPREPROCESSED', help='Raw data files directory. (Matlab extension .mat)')
parser.add_argument('-pdd', '--processed_data_dir', type=str, default='../data/processed_data', help='Where to put processed data files from DEAPDataset')
parser.add_argument('-ef', '--eeg_feature', type=str, default='wav', choices=['wav','psd','raw'], help='Feature extraction method for EEG signals')
parser.add_argument('-t', '--target', type=str, default='participant_id', choices=['participant_id','video_id','emotions_combined','valence','arousal','dominance','liking'], help='Target for samples')
parser.add_argument('-p', '--participant', type=int, default=None, choices=list(range(1,33)), help='If != None it specifies an individual participant to use as raw data')
parser.add_argument('-rgc','--remove_global_connections', default=False, action='store_true',help='Remove global connections from the graph adjacency matrix')
parser.add_argument('-nts', '--number_train_samples', type=int, default=8, help='Number of training samples per class')
parser.add_argument('-dsd', '--dont_shuffle_data', default=False, action='store_true', help='Do not shuffle dataset before sub sampling ')
parser.add_argument('-dev', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda','cpu'], help='Device')
parser.add_argument('-dt', '--dont_train', default=False, action='store_true', help='Load checkpoint and test')
parser.add_argument('-bs', '--batch_size', type=int, default=16)
parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Print training logs to console')
parser.add_argument('-kfvo', '--kfold_validation_offset', type=int, default=0)
parser.add_argument('-ws', '--window_size', type=float, default=1.0, help='Size of the windows taken from each video (seconds). Should be divisible by 60')


# Train args
parser.add_argument('-m', '--model', type=str, default='MLP', choices=['MLP','CNN','GraphConv','GIN'], help='Which model architecture to train')
parser.add_argument('-hc', '--hidden_channels', type=int, default=64, help='Number of hidden channels')
parser.add_argument('-opt', '--optimizer', type=str, default='Adam', choices=['Adam','Adagrad','SGD'])
parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
parser.add_argument('-dr', '--dropout_rate', type=float, default=0.4)
parser.add_argument('-lrd', '--learning_rate_decay', type=float, default=0)
parser.add_argument('-wd', '--weight_decay', type=float, default=0)
parser.add_argument('-esp', '--early_stopping_patience', type=int, default=50)
parser.add_argument('-st', '--shuffle_train', default=False, action='store_true', help='Shuffle train dataloader')
parser.add_argument('-me', '--max_epoch', type=int, default=10000)
parser.add_argument('-act', '--activation_funct', type=str, default='relu', choices=['relu','tanh'])


# Test args
parser.add_argument('-tmd', '--test_model_dict', type=str, default='checkpoint1', help='Model to test')
parser.add_argument('-trd', '--test_results_dir', type=str, default='log1')
parser.add_argument('-wtr', '--write_test_results', default=False, action='store_true', help='Log results to csv')


args = parser.parse_args()


dataset = DEAPDataset(args)
print(dataset)

train_mask, val_mask, test_mask = get_split_indices(args.target, args.number_train_samples, len(dataset) ,args.dont_shuffle_data)

samples_per_participant = int((60/args.window_size)*40)

train_dataset = dataset[train_mask]
val_dataset = dataset[val_mask]
test_dataset = dataset[test_mask]

train_samples = []
for i in range(args.number_train_samples):
    idx = train_mask[32*i]
    video = int(idx//(samples_per_participant/40))
    second = idx/(samples_per_participant/40)*60%60
    print(f'Picking train sample {idx}/{samples_per_participant} (Video {video} seconds {second}-{second+args.window_size}) - ({args.eeg_feature})')
    train_samples.append({'idx':idx, 'samples_per_participant':samples_per_participant, 'video':video, 'second_f':second, 'second_t':second+args.window_size})

    # Plot feature matrices
    if False:
        for j in range(32):
            plt.subplot(1, 32, j+1)
            plt.imshow(train_dataset[i*32+j].x)
            plt.axis('off')
        plt.show()

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
    model = MLP(in_channels,args.hidden_channels, n_classes, args.dropout_rate,args.activation_funct).to(args.device) 
elif args.model == 'CNN':
    model = CNN(in_channels,args.hidden_channels, n_classes, args.dropout_rate,args.activation_funct).to(args.device) 
elif args.model == 'GraphConv':
    model = GraphConv(in_channels,args.hidden_channels, n_classes, args.dropout_rate,args.activation_funct).to(args.device) 
elif args.model == 'GIN':
    model = GIN(in_channels,args.hidden_channels, n_classes, args.dropout_rate,args.activation_funct).to(args.device) 

print(model)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'Model parameter count: {pytorch_total_params}')
# print(f'K fold offset: {args.kfold_validation_offset}')
print(f'Train dataset: {train_dataset} | Validation dataset: {val_dataset} | Test dataset: {test_dataset}',)

train_time, best_epoch = None, None
if not args.dont_train:
    train_time, best_epoch = train_main(model,train_dataset,val_dataset,criterion,args)

test_main(model,test_dataset,train_dataset,val_dataset,criterion,args,train_time, best_epoch, train_samples)

