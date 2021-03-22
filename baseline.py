#!/usr/bin/env python

import torch
import numpy as np
import torch.nn.functional as F
from DEAPDataset import DEAPDataset, train_val_test_split

def baseline(args):
    ROOT_DIR = './'
    RAW_DIR = 'data/matlabPREPROCESSED'
    PROCESSED_DIR = 'data/graphProcessedData'

    dataset = DEAPDataset(root= ROOT_DIR, raw_dir= RAW_DIR, processed_dir=PROCESSED_DIR,args=args)

    train_set, val_set, test_set = train_val_test_split(dataset)
    train_set = train_set+val_set


    labels = [g.y for g in train_set]
    labels = torch.stack(labels)

    # Predictions are sampled from the mean. Input independent.
    predictions = labels.mean(dim=0)

    mses = []
    l1s = []
    for data in test_set:
        print('Prediction: ')
        print(predictions.detach().numpy())
        print('Ground truth: ')
        print(data.y.detach().numpy())
        print(f'Mean average error: {F.l1_loss(predictions,data.y).item()}')
        l1 = F.l1_loss(predictions,data.y).item()
        mse = F.mse_loss(predictions,data.y).item()
        print(f'Mean squared error: {mse}')
        print(f'Mean average error: {l1}')
        mses.append(mse)
        l1s.append(l1)
    print('----------------')
    print(f'MEAN AVERAGE ERROR FOR TEST SET: {np.array(l1).mean()}')
    print(f'MEAN SQUARED ERROR FOR TEST SET: {np.array(mses).mean()}')
