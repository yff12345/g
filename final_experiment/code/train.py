import torch
import time
import numpy as np
from util import get_split_indices
from test import test_epoch
from torch_geometric.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter
from metrics import accuracy_metric, f1_metric, precision_metric, recall_metric, roc_metric
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


# learning_rate = 1e-5
# lrs, losses2 = [], []

def train_epoch(model, loader ,optimizer ,criterion,args):

    # global learning_rate
    # global lrs
    # global losses

    model.train()
    losses, outputs, targets = [], [] , []
    for batch in loader:
        # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay=args.learning_rate_decay)
        optimizer.zero_grad()
        batch = batch.to(args.device)
        y = batch.y
        out = model(batch)
        outputs.append(out.detach())
        targets.append(y)
        loss = criterion(out,y)
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()


        # lrs.append(learning_rate)
        # losses2.append(loss.detach().item())
        # learning_rate += learning_rate * 0.02
    return np.array(losses).mean(),torch.cat(outputs), torch.cat(targets)

def main(model,train_dataset,val_dataset,criterion, args):


    # global learning_rate
    # global lrs
    # global losses

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Optimizers
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate, weight_decay=args.learning_rate_decay)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(),lr=args.learning_rate, lr_decay=args.learning_rate_decay, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate, weight_decay=args.weight_decay)

    # Train
    best_val_loss, early_stopping_count, best_epoch = np.inf, 0, 0
    print('Training...')
    start_time = time.time()
    for epoch in range(1,args.max_epoch+1):
        # Train epoch
        mean_train_loss, train_outputs, train_targets = train_epoch(model, train_loader ,optimizer ,criterion,args)
        mean_val_loss, val_outputs, val_targets = test_epoch(model, val_loader ,criterion,args)

        # Metrics
        train_acc = accuracy_metric(train_outputs, train_targets)
        val_acc = accuracy_metric(val_outputs, val_targets)
        train_prec = precision_metric(train_outputs, train_targets)
        val_prec = precision_metric(val_outputs, val_targets)
        train_reca = recall_metric(train_outputs, train_targets)
        val_reca = recall_metric(val_outputs, val_targets)
        train_f1 = f1_metric(train_outputs, train_targets)
        val_f1 = f1_metric(val_outputs, val_targets)
        train_roc = roc_metric(train_outputs, train_targets)
        val_roc = roc_metric(val_outputs, val_targets)

        # Log to console
        if args.verbose:
            print(f'--Epoch : {epoch} --')
            # print(learning_rate)
            print(f'Train loss: {mean_train_loss:.5f} - Val loss: {mean_val_loss:.5f} ')
            print(f'Train acc: {train_acc:.3f} - Val acc: {val_acc:.3f} ')
            print(f'Train precision: {train_prec:.3f} - Val precision: {val_prec:.3f} ')
            print(f'Train recall: {train_reca:.3f} - Val recall: {val_reca:.3f}')
            print(f'Train F1: {train_f1:.3f} - Val F1: {val_f1:.3f}')
            print(f'Train auc: {train_roc:.3f} - Val auc: {val_roc:.3f}\n')

        # Early stopping and checkpoint
        if mean_val_loss < best_val_loss-0.0001:
            best_epoch = epoch
            best_val_loss = mean_val_loss
            early_stopping_count = 0
            torch.save(model.state_dict(),f'../checkpoints/{args.test_model_dict}') 
        else:
            early_stopping_count += 1
            if early_stopping_count >= args.early_stopping_patience:
                break

        # if learning_rate>=1e-1:
        #     # fig, ax = plt.subplots()
        #     # plt.subplot(1, 3, 1)
        #     # plt.xlabel('batch number')
        #     # plt.ylabel('learning rate')
        #     # plt.yscale('log')
        #     # plt.plot(lrs)
        #     # plt.subplot(1, 3, 2)
        #     # plt.xlabel('batch number')
        #     # plt.ylabel('loss')
        #     # plt.plot(losses2)
        #     # plt.subplot(1, 3, 3)
        #     plt.title(f'{args.model} ({args.hidden_channels} hidden channels)')
        #     plt.xlabel('learning rate')
        #     plt.ylabel('loss (smoothed)')
        #     plt.xscale('log')
        #     plt.plot(lrs,gaussian_filter1d(losses2, sigma=2))
        #     plt.show()
        #     exit()

    end_time = time.time()
    print('--Finished training--')

    train_time = end_time-start_time

    print(f'Training time (s): { train_time:.2f}')
    print(f'Total epochs: { epoch } ; Model picked from epoch {best_epoch}')
    print(f'Average time per epoch: {(train_time / epoch):.2f}')

    return train_time, best_epoch