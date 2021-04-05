#!/usr/bin/env python

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from models.GNNLSTM import GNNLSTM
from models.STGCN.STGCN import STGCN
from matplotlib import pyplot as plt
from tqdm import tqdm

# Define training and eval functions for each epoch (will be shared for all models)
def train_epoch(model,loader,optim,criterion,device,target,args):
    target = {'valence':0,'arousal':1,'dominance':2,'liking':3}[target]
    if model.eval_patience_reached:
      return -1
    model.train()
    metrics = {
      # Main loss as calculated from criterion
      "loss":[]
    }
    for batch in tqdm(loader):
      optim.zero_grad()
      batch = batch.to(device)
      out = model(batch,args)
      # Gets first label for every graph
      target_label = batch.y.T[target].unsqueeze(1)
      loss = criterion(out, target_label)
      # Update parameters
      loss.backward()
      optim.step()
      # Record metrics
      metrics["loss"].append(loss.item())
    mean_epoch_loss = np.array(metrics["loss"]).mean()
    model.train_losses.append(mean_epoch_loss)
    return mean_epoch_loss

def eval_epoch(model,loader,device,target,args,criterion,epoch=-1, model_is_training = False, early_stopping_patience = None):
    target = {'valence':0,'arousal':1,'dominance':2,'liking':3}[target] 
    if model.eval_patience_reached and model_is_training:
      return -1,-1
    model.eval()
    metrics = {
      # Main loss as calculated from criterion
      "loss":[],
      "l1":[],
      "mse":[]
    }
    # Evaluation
    for batch in loader:
      batch = batch.to(device) 
      out = model(batch,args)
      print(out.detach().cpu())
      target_label = batch.y.T[target].unsqueeze(1)
      metrics["loss"].append(criterion(out, target_label).item())
      metrics["mse"].append(F.mse_loss(out,target_label).item())
      metrics["l1"].append(F.l1_loss(out,target_label).item())
    loss = np.array(metrics["loss"]).mean()
    
    # Early stopping and checkpoint
    if model_is_training:
      model.eval_losses.append(loss)
      # Save current best model locally
      if loss < model.best_val_loss:
        model.best_val_loss = loss
        model.best_epoch = epoch
        torch.save(model.state_dict(),f'./best_params_{target}') 
        model.eval_patience_count = 0
      # Early stopping
      elif args.early_stopping_patience is not None:
          model.eval_patience_count += 1
          if model.eval_patience_count >= args.early_stopping_patience:
            model.eval_patience_reached = True
    return loss, np.array(metrics["mse"]).mean(), np.array(metrics["l1"]).mean()


def train (args, train_data_in, device):

  BATCH_SIZE = args.batch_size

  # Define loss function 
  # criterion = torch.nn.MSELoss()
  criterion = torch.nn.BCELoss() if not args.regression_labels else torch.nn.MSELoss()

  # Define model targets. Each target has a model associated to it.
  # Train 1 target at a time
  targets = ['valence','arousal','dominance','liking']

  target = targets[args.n_targets-1]
  print(f'Training {target} model...')

  # Print losses over time (train and val)
  plt.figure(figsize=(10, 10)) # TODO: fix

  # Train models one by one as opposed to having an array [] of models. Avoids CUDA out of memory error
  model = STGCN(window_size=128).to(device)

  # print(sum(p.numel() for p in model.parameters()))
  # exit()

  # Define optimizer
  # optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.l2_reg_beta, amsgrad=False)
  # optim = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
  # optim = torch.optim.SGD(model.parameters(),lr=args.learning_rate, momentum= 0.7, weight_decay=args.l2_reg_alpha)
  # optim = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate, lr_decay=0.001, weight_decay=args.l2_reg_alpha, initial_accumulator_value=0, eps=1e-10)
  optim = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate, rho=0.9, eps=1e-06, weight_decay=args.l2_reg_alpha)
  # optim = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.99, eps=1e-08, weight_decay=args.l2_reg_alpha, momentum=0, centered=False)

  # Instantiate optimizer
  scheduler = StepLR(optim, step_size=30, gamma=args.scheduler_gamma)

  # Number of slices (7 of 5 videos each)
  k_fold_splits = 7
  k_fold_size = int(int(35/k_fold_splits) * len(train_data_in) / 35)

  for epoch in range(args.max_epoch):
    if args.dont_kfold_validation:
      assert args.batch_size == 1
      # 'TODO: Disable kfold with bs != 1'
      val_data = train_data_in[30:]
      train_data = train_data_in[:30]
    else:
      # KFOLD train/val split (30/5)
      val_data = train_data_in[(epoch%k_fold_splits)*k_fold_size:(epoch%k_fold_splits+1)*k_fold_size]
      a = train_data_in[0:(epoch%k_fold_splits)*k_fold_size]
      b = train_data_in[(epoch%k_fold_splits+1)*k_fold_size:]
      print(f'a length : {len(a)}')
      print(f'b length : {len(b)}')
      train_data = a + b
      print(f'Val data length : {len(val_data)}')
      print(f'Train data length : {len(train_data)}')
    # Create loaders for epoch
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=not args.dont_shuffle_train)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
   
    # Train epoch for every model
    t_e_loss = train_epoch(model,train_loader,optim,criterion,device,target=target,args=args)
    # Validation epoch for every model
    v_e_metrics = eval_epoch(model,val_loader,device,target,args,criterion,epoch,model_is_training = True) 
    # Break if model has reached patience limit. Model parameters are saved to 'best_params' file.
    if t_e_loss == -1:
      break
    # Epoch results
    print(f'------ Epoch {epoch} ------')
    print(f'{target}: Train loss: {t_e_loss:.2f} | Val loss: {v_e_metrics[0]:.2f} | Val mse: {v_e_metrics[1]:.2f} | Val l1: {v_e_metrics[2]:.2f}')
    # Gradually reduce lr
    scheduler.step()
  
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
  be = model.best_epoch
  model.load_state_dict(torch.load(f'./best_params_{args.n_targets-1}'))
  # Evaluating best models
  final_eval = eval_epoch(model,val_loader,device,target,args,criterion,epoch,model_is_training = False) 
  print (f'{target} (epoch {be}): Validation mse: {final_eval[2]:.2f}')
