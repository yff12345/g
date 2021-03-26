#!/usr/bin/env python

import torch
import argparse
from train import train
from test import test
from baseline import baseline

from DEAPDataset import DEAPDataset, describe_graph


from util import train_test_split

# ARGUMENTS
# Define argument parser
parser = argparse.ArgumentParser(description='Process eeg emotions. http://www.eecs.qmul.ac.uk/mmv/datasets/deap/')
# Common args
parser.add_argument('-t', '--n_targets', choices=[1,2,3,4], type=int, default=1,
                    help='1:Valence, 2:Valence,Arousal, 3:Valence,Arousal,Dominance 4:Valence,Arousal,Dominance,Liking')
parser.add_argument('-st','--single_target', default=False, action='store_true',
                    help='Use invididual target (-t)')
parser.add_argument('-pf', '--participant_from', choices=range(1,33), type=int, default=1,
                    help='Which participant data to be used')
parser.add_argument('-pt', '--participant_to', choices=range(1,33), type=int, default=1,
                    help='Which participant data to be used')
parser.add_argument('-bs', '--batch_size', type=int, default=1,
                    help='Batch size')
parser.add_argument('-me', '--max_epoch', type=int, default=100,
                    help='Max epochs for training')
parser.add_argument('-gc','--global_connections', default=True, action='store_false',help='Add global connections to the graph adjacency matrix')
parser.add_argument('-cl','--classification_labels', default=False, action='store_true',
                    help='Use high[>5],low[<5] emotion labels for classification instead of regression')
# Train logic args
parser.add_argument('-dst','--dont_shuffle_train', default=False, action='store_true')
parser.add_argument('-esp', '--early_stopping_patience', type=int, default=3,
                    help='Early stopping patience (epochs)')
parser.add_argument('-l1', '--l1_reg_alpha',  type=float, default=0,
                    help='l1 regularization')
parser.add_argument('-l2', '--l2_reg_alpha',  type=float, default=0,
                    help='l2 regularization')
parser.add_argument('-lr', '--learning_rate',  type=float, default=0.001,
                    help='learning rate')
# Test logic args
parser.add_argument('--test', default=False, action='store_true')
# Baseline logic
parser.add_argument('--baseline', default=False, action='store_true')
# Model specific
parser.add_argument('-vc','--visualize_convs', default=False, action='store_true')

# Retrieve args from terminal
args = parser.parse_args()
# Validate arguments
assert args.participant_to >= args.participant_from

# COMMON LOGIC
ROOT_DIR = './'
RAW_DIR = 'data/matlabPREPROCESSED'
PROCESSED_DIR = 'data/graphProcessedData'

dataset = DEAPDataset(root= ROOT_DIR, raw_dir= RAW_DIR, processed_dir=PROCESSED_DIR,args=args)

# 35 training videos (30/5) and  5 test
train_data, test_data = train_test_split(dataset,35)
# Describe graph structure (same for all instances)
describe_graph(train_data[0])


 # Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')



if args.test:
    test(args,test_data,device)
    pass
elif args.baseline:
    baseline(args,train_data,test_data)
    pass
else:
    train(args,train_data, device)
    pass



