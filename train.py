#!/usr/bin/env python

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from DEAPDataset import DEAPDataset, train_test_split, plot_graph, describe_graph, plot_graph
from models.GNNLSTM import GNNLSTM
from matplotlib import pyplot as plt
from tqdm import tqdm

# Define training and eval functions for each epoch (will be shared for all models)
def train_epoch(model,loader,optim,criterion,device,target,args):
    target = {'valence':0,'arousal':1,'dominance':2,'liking':3}[target]
    if model.eval_patience_reached:
      return -1, -1
    model.train()
    epoch_losses = []
    for batch in tqdm(loader):
      optim.zero_grad()
      batch = batch.to(device)
      out = model(batch)
      # Gets first label for every graph
      target_label = batch.y.T[target].unsqueeze(1)
      mse_loss = criterion(out, target_label)
      # REGULARIZATION
      l1_regularization, l2_regularization = torch.tensor(0, dtype=torch.float).to(device), torch.tensor(0, dtype=torch.float).to(device)
      for param in model.parameters():
        l1_regularization += (torch.norm(param, 1)**2).float()
        l2_regularization += (torch.norm(param, 2)**2).float()
      # loss = mse_loss + args.l2_reg_beta * l2_regularization + args.l1_reg_alpha * l1_regularization
      loss = mse_loss
      loss.backward()
      optim.step()
      epoch_losses.append(loss.item())
    epoch_mean_loss = np.array(epoch_losses).mean()
    return epoch_mean_loss, mse_loss

def eval_epoch(model,loader,device,target,args,epoch=-1, model_is_training = False, early_stopping_patience = None):
    target = {'valence':0,'arousal':1,'dominance':2,'liking':3}[target] 
    if model.eval_patience_reached and model_is_training:
      return [-1,-1]
    model.eval()
    mses,l1s = [],[]
    # Evaluation
    for batch in loader:
      batch = batch.to(device) 
      out = model(batch)
      target_label = batch.y.T[target].unsqueeze(1)
      print(out.detach().cpu())
      mses.append(F.mse_loss(out,target_label).item())
      l1s.append(F.l1_loss(out,target_label).item())
    e_mse, e_l1 = np.array(mses).mean(), np.array(l1s).mean()
    l1_regularization, l2_regularization = torch.tensor(0, dtype=torch.float).to(device), torch.tensor(0, dtype=torch.float).to(device)
    for param in model.parameters():
      l1_regularization += (torch.norm(param, 1)**2).float()
      l2_regularization += (torch.norm(param, 2)**2).float()
    # loss = e_mse + args.l2_reg_beta * l2_regularization + args.l1_reg_alpha * l1_regularization
    loss = e_mse
    # Early stopping and checkpoint
    if model_is_training:
      model.eval_losses.append(loss)
      # Save current best model locally
      if loss < model.best_val_mse:
        model.best_val_mse = loss
        model.best_epoch = epoch
        torch.save(model.state_dict(),f'./best_params_{target}') # This slows everything
        model.eval_patience_count = 0
      # Early stopping
      elif args.early_stopping_patience is not None:
          model.eval_patience_count += 1
          if model.eval_patience_count >= args.early_stopping_patience:
            model.eval_patience_reached = True
    return loss, e_l1, e_mse


def train (args):
  ROOT_DIR = './'
  RAW_DIR = 'data/matlabPREPROCESSED'
  PROCESSED_DIR = 'data/graphProcessedData'
  # Initialize dataset
  dataset = DEAPDataset(root= ROOT_DIR, raw_dir= RAW_DIR, processed_dir=PROCESSED_DIR,args=args)
  # 30 samples are used for training, 5 for validation and 5 are saved for testing
  train_set, _ = train_test_split(dataset)            
  # Describe graph structure (same for all instances)
  describe_graph(train_set[0])
  # Set batch size
  BATCH_SIZE = args.batch_size

  # Use GPU if available
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Device: {device}')
  # Define loss function 
  criterion = torch.nn.MSELoss()
  # Define model targets. Each target has a model associated to it.
  # Train 1 target at a time
  target = ['valence','arousal','dominance','liking'][args.n_targets-1]
  
  print(f'Training {target} model...')

  # MODEL PARAMETERS
  # in_channels = train_set[0].num_node_features

  # Print losses over time (train and val)
  plt.figure(figsize=(10, 10))
  # Train models one by one as opposed to having an array [] of models. Avoids CUDA out of memory error
  model = GNNLSTM().to(device)
  # optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.l2_reg_beta, amsgrad=False)
  # optim = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
  # optim = torch.optim.SGD(model.parameters(),lr=args.learning_rate, momentum= 0.7)
  # optim = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0.001, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
  # optim = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=args.l2_reg_beta)
  optim = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=args.l2_reg_beta, momentum=0, centered=False)

  k_fold_splits = 7
  k_fold_size = 5   
  for epoch in range(args.max_epoch):
    # KFOLD train/val split (30/5)
    val_data = train_set[(epoch%k_fold_splits)*k_fold_size:(epoch%k_fold_splits+1)*k_fold_size]
    a = train_set[0:(epoch%k_fold_splits)*k_fold_size]
    b = train_set[(epoch%k_fold_splits+1)*k_fold_size:]
    train_data = a + b
    # Create loaders for epoch
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=not args.dont_shuffle_train)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
   
    # Train epoch for every model
    t_e_loss = train_epoch(model,train_loader,optim,criterion,device,target=target,args=args)
    # Validation epoch for every model
    v_e_loss = eval_epoch(model,val_loader,device,target,args,epoch,model_is_training = True) 
    # Break if model has reached patience limit. Model parameters are saved to 'best_params' file.
    if t_e_loss[0] == -1:
      break
    # Epoch results
    print(f'------ Epoch {epoch} ------')
    print(f'{target}: Train loss: {t_e_loss[0]:.2f} | Train mse: {t_e_loss[1]:.2f} | Val loss: {v_e_loss[0]:.2f} | Val mse: {v_e_loss[2]:.2f}')
  
  # plt.subplot(2,2,i+1)
  plt.plot(model.train_losses)
  plt.plot(model.eval_losses)
  plt.title(f'{target} losses')
  plt.ylabel('loss (mse)')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper right')
  plt.savefig(f'train_loss_{target}.png')

  # Final evaluation
  print(f'------ Final model eval ------ \n')

  model.load_state_dict(torch.load(f'./best_params_{args.n_targets-1}'))
  # Evaluating best models
  final_eval = eval_epoch(model, val_loader,device,model_is_training = False,target=target,args=args)
  print (f'{target} (epoch {model.best_epoch}): Validation mse: {final_eval[2]:.2f}')
