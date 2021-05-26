import torch
import time
import numpy as np
from util import get_split_indices
from test import test_epoch
from torch_geometric.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter
from metrics import accuracy_metric, f1_metric, precision_metric, recall_metric


def train_epoch(model, loader ,optimizer ,criterion,args):
    model.train()
    losses, outputs, targets = [], [] , []
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(args.device)
        y = batch.y if args.target != 'emotion_labels' else (batch.y[:,args.target_emotion] > 5).long()
        out = model(batch)
        outputs.append(out)
        targets.append(y)
        # print(out)
        loss = criterion(out,y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return np.array(losses).mean(),torch.cat(outputs), torch.cat(targets)

def main(model,dataset,criterion, args):

    train_mask,val_mask = get_split_indices(args.target,args.number_validation_targets,len(dataset))

    # Split training dataset into training and validation
    train_dataset = dataset[train_mask]
    val_dataset = dataset[val_mask]

    print(f'Train dataset: {train_dataset} | Validation dataset: {val_dataset}',)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate, weight_decay=args.learning_rate_decay)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(),lr=args.learning_rate, lr_decay=args.learning_rate_decay, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate, weight_decay=args.weight_decay)

    writer = SummaryWriter()

    start_time = time.time()

    best_val_loss = np.inf
    early_stopping_count = 0
    print('Training...')
    for epoch in range(1,args.max_epoch+1):
        # Train epoch
        mean_train_loss, train_outputs, train_targets = train_epoch(model, train_loader ,optimizer ,criterion,args)
        mean_val_loss, val_outputs, val_targets = test_epoch(model, val_loader ,criterion,args)
        # Accuracy
        train_acc = accuracy_metric(train_outputs, train_targets)
        val_acc = accuracy_metric(val_outputs, val_targets)
        # F1
        train_f1 = f1_metric(train_outputs, train_targets)
        val_f1 = f1_metric(val_outputs, val_targets)
        # Precision
        train_prec = precision_metric(train_outputs, train_targets)
        val_prec = precision_metric(val_outputs, val_targets)
        # Recall
        train_reca = recall_metric(train_outputs, train_targets)
        val_reca = recall_metric(val_outputs, val_targets)

        # Report metrics
        writer.add_scalar('Loss/train', mean_train_loss, epoch)
        writer.add_scalar('Loss/test', mean_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', val_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/test', val_f1, epoch)
        writer.add_scalar('Precision/train', train_prec, epoch)
        writer.add_scalar('Precision/test', val_prec, epoch)
        writer.add_scalar('Recall/train', train_reca, epoch)
        writer.add_scalar('Recall/test', val_reca, epoch)
        if args.verbose:
            print(f'--Epoch : {epoch} --')
            print(f'Train loss: {mean_train_loss:.5f} - Val loss: {mean_val_loss:.5f} ')
            print(f'Train acc: {train_acc:.3f} - Val acc: {val_acc:.3f} ')
            print(f'Train F1: {train_f1:.3f} - Val F1: {val_f1:.3f} ')
            print(f'Train precision: {train_prec:.3f} - Val precision: {val_prec:.3f} ')
            print(f'Train recall: {train_reca:.3f} - Val recall: {val_reca:.3f} \n')

        # Early stopping and checkpoint
        if mean_val_loss < best_val_loss-0.0001:
            best_val_loss = mean_val_loss
            early_stopping_count = 0
            torch.save(model.state_dict(),'./best_params_tmp') 
        else:
            early_stopping_count += 1
            if early_stopping_count >= args.early_stopping_patience:
                break


    end_time = time.time()
    print('--Finished training--')


    train_time = end_time-start_time

    print(f'Training time (s): { train_time:.2f}')
    print(f'Total epochs: { epoch }')
    print(f'Average time per epoch: {(train_time / epoch):.2f}')