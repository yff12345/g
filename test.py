#!/usr/bin/env python

import torch
import numpy as np
import torch.nn.functional as F
from models.GNNLSTM import GNNLSTM
from DEAPDataset import DEAPDataset, train_val_test_split
from torch_geometric.data import DataLoader
import itertools
np.set_printoptions(precision=2)

def test(args):

  if args.single_target:
    targets = targets[args.n_targets]
  else:
    targets = targets[:args.n_targets]




  ROOT_DIR = './'
  RAW_DIR = 'data/matlabPREPROCESSED'
  PROCESSED_DIR = 'data/graphProcessedData'
  dataset = DEAPDataset(root= ROOT_DIR, raw_dir= RAW_DIR, processed_dir=PROCESSED_DIR,args=args)
  # 5 testing samples per participant (30/5/5)
  _, _, test_set = train_val_test_split(dataset)

  test_loader = DataLoader(test_set, batch_size=args.batch_size)

   # MODEL PARAMETERS
  # in_channels = test_set[0].num_node_features

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Device: {device}')

  if args.all_targets:
    targets = ['valence','arousal','dominance','liking']
  else:
    targets = ['valence','arousal','dominance','liking'][args.n_targets-1:args.n_targets]

  target_index = {'valence':0,'arousal':1,'dominance':2,'liking':3}
  models = [GNNLSTM().to(device).eval() for target in targets]

  # Load best performing params on validation
  for i,target in enumerate(targets):
    index = target_index[target]
    models[i].load_state_dict(torch.load(f'./best_params_{index}'))

  mses,l1s,accs = [],[],[]
  for batch in test_loader:
    batch = batch.to(device)
    predictions = [model(batch) for model in models]
    if args.all_targets:
      target = batch.y.narrow(1,0,len(targets)).view(-1)
      predictions = torch.stack(predictions,dim=1).view(-1)
    else:
      target = batch.y.narrow(1,0,len(targets)).view(-1)
      predictions = torch.stack(predictions,dim=1).view(1)
    
    right = (target > 5) == (predictions > 5)
    acc = (right.sum()/len(target)).item()
    accs.append(acc)
    
    print('-Predictions-')
    print(predictions.cpu().detach().numpy(),'\n')
    print('-Ground truth-')
    print(target.cpu().detach().numpy(),'\n')
    mse = F.mse_loss(predictions,target).item()
    l1 = F.l1_loss(predictions,target).item()
    mses.append(mse)
    l1s.append(l1)
    print(f'Mean average error: {l1}')
    print(f'Mean squared error: {mse}')
    print(f'Accuracy: {acc*100}%')

  print('----------------')
  print(f'MEAN AVERAGE ERROR FOR TEST SET: {np.array(l1s).mean()}')
  print(f'MEAN SQUARED ERROR FOR TEST SET: {np.array(mses).mean()}')
  print(f'MEAN ACCURACY: {np.array(accs).mean()}')