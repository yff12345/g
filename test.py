#!/usr/bin/env python

import torch
import numpy as np
import torch.nn.functional as F
from models.GNNLSTM import GNNLSTM
from DEAPDataset import DEAPDataset
from torch_geometric.data import DataLoader
np.set_printoptions(precision=2)

def test(args, test_data_in, device):
  targets = ['valence','arousal','dominance','liking']
  if args.single_target:
    targets = [targets[args.n_targets-1]]
  else:
    targets = targets[:args.n_targets]
    

  test_loader = DataLoader(test_data_in, batch_size=args.batch_size)

  target_index = {'valence':0,'arousal':1,'dominance':2,'liking':3}

  models = [GNNLSTM().to(device).eval() for target in targets]

  # Load best performing params on validation
  for i,target in enumerate(targets):
    index = target_index[target]
    models[i].load_state_dict(torch.load(f'./best_params_{index}'))

  metrics = {
      "mse":[],
      "l1":[],
      "acc":[]
    }
  for batch in test_loader:
    batch = batch.to(device)
    predictions = [model(batch) for model in models]
    predictions = torch.stack(predictions,dim=1).view(-1,len(targets))
    # target = batch.y.narrow(1,0,len(targets)).view(-1)

    if args.single_target:
      target = batch.y[:,args.n_targets-1].view(-1,1)
    else:
      target = batch.y.narrow(1,0,len(targets)).view(-1,len(targets))

    print(f'Predictions:\n {predictions.cpu().detach().numpy()}')
    print(f'Target (gt):\n {target.cpu().detach().numpy()}')
      
    
    right = (target > 5) == (predictions > 5)
    
    mse = F.mse_loss(predictions,target).item()
    l1 = F.l1_loss(predictions,target).item()
    acc = (right.sum()/(len(target)*args.batch_size)).item()    
    metrics["mse"].append(mse)
    metrics["l1"].append(l1)
    metrics["acc"].append(acc)
    print(f'Mean squared error: {mse}')
    print(f'Mean average error: {l1}')
    print(f'Accuracy: {acc*100}%')

  print('----------------')
  print(f'MEAN SQUARED ERROR FOR TEST SET: {np.array(metrics["mse"]).mean()}')
  print(f'MEAN AVERAGE ERROR FOR TEST SET: {np.array(metrics["l1"]).mean()}')
  print(f'MEAN ACCURACY: {np.array(metrics["acc"]).mean()}')