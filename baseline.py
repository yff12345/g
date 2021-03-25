#!/usr/bin/env python

import torch
import numpy as np
import torch.nn.functional as F


def baseline(args,train_data,test_data):
    
    labels = [g.y for g in train_data]
    labels = torch.stack(labels)

    # Predictions are sampled from the mean. Input independent.
    predictions = labels.mean(dim=0)

    metrics = {
        "mse":[],
        "l1":[]
    }

    for data in test_data:
        print(f'Ground truth: {print(data.y.detach().numpy())}')
        
        mse = F.mse_loss(predictions,data.y).item()
        metrics["mse"].append(mse)
        print(f'Mean squared error: {mse}')
        
        l1 = F.l1_loss(predictions,data.y).item()
        metrics["l1"].append(l1)
        print(f'Mean average error: {l1}')

        
    print(f'Predictions (Same for all test instances):{predictions}')
    print('----------------')
    print(f'MEAN AVERAGE ERROR FOR TEST SET: {np.array(metrics["l1"]).mean()}')
    print(f'MEAN SQUARED ERROR FOR TEST SET: {np.array(metrics["mse"]).mean()}')
