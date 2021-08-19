
import torch
import os
import csv   
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import  DataLoader
from metrics import accuracy_metric, f1_metric, precision_metric, recall_metric, roc_metric


# Visualize latent space
from sklearn.manifold import TSNE

def test_epoch(model, loader ,criterion,args):
    model.eval()
    losses, outputs, targets = [], [] , [] # This might waste GPU space during testing
    confusion_matrix = torch.zeros(32, 32)
    for batch in loader:
        batch = batch.to(args.device)
        y = batch.y if args.target != 'emotion_labels' else (batch.y[:,args.target_emotion] > 5).long()
        out = model(batch)
        outputs.append(out.detach())
        targets.append(y)
        loss = criterion(out,y)
        losses.append(loss.detach().item())

        _, preds = torch.max(out, 1)
        for t, p in zip(y.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
    # plt.imshow(confusion_matrix)
    # plt.show()


    return np.array(losses).mean(),torch.cat(outputs), torch.cat(targets)

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

    print(confusion_matrix)

def main(model, dataset,train_dataset,val_dataset, criterion , args, train_time, best_epoch, train_samples):
    # Load checkpoint
    model.load_state_dict(torch.load(f'../checkpoints/{args.test_model_dict}'))

    # Create loader
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) 

    # Test epoch
    mean_test_loss, test_outputs, test_targets = test_epoch(model, test_loader ,criterion,args)

    # Metrics
    test_acc = accuracy_metric(test_outputs, test_targets)
    test_f1 = f1_metric(test_outputs, test_targets)
    test_prec = precision_metric(test_outputs, test_targets)
    test_reca = recall_metric(test_outputs, test_targets)
    test_roc = roc_metric(test_outputs, test_targets)
    
    # Log to console
    print(f'\n--Testing--')
    print(f'Test loss: {mean_test_loss:.5f}')
    print(f'Test acc: {test_acc:.3f}  ')
    print(f'Test F1: {test_f1:.3f} ')
    print(f'Test precision: {test_prec:.3f}')
    print(f'Test recall: {test_reca:.3f}')
    print(f'Test ROC auc: {test_roc:.3f} \n')

    # Train epoch
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    mean_train_loss, train_outputs, train_targets = test_epoch(model, train_loader ,criterion,args)
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
    print(f'Train loss: {mean_train_loss:.5f} - Val loss: {mean_val_loss:.5f} ')
    print(f'Train acc: {train_acc:.3f} - Val acc: {val_acc:.3f} ')
    print(f'Train precision: {train_prec:.3f} - Val precision: {val_prec:.3f} ')
    print(f'Train recall: {train_reca:.3f} - Val recall: {val_reca:.3f}')
    print(f'Train F1: {train_f1:.3f} - Val F1: {val_f1:.3f}')
    print(f'Train auc: {train_roc:.3f} - Val auc: {val_roc:.3f}\n')

    # Visualize latent space (?) -> Needs more testing
    # out_np = test_outputs.detach().cpu().numpy()
    # target_np = test_targets.detach().cpu().numpy()
    # z = TSNE(n_components=2).fit_transform(out_np)
    # plt.figure(figsize=(10,10))
    # plt.xticks([])
    # plt.yticks([])

    # plt.scatter(z[:, 0], z[:, 1], s=70, c=target_np, cmap="Set2", label=target_np)
    # plt.show()



    # Write results to CSV
    if args.write_test_results:
        print('Writing to logs...')
        records = {}
        # Define records
        settings = ['eeg_feature','model','hidden_channels','window_size','batch_size','learning_rate','dropout_rate','weight_decay', 'activation_funct','number_train_samples']
        for s in settings:
            records[s] = getattr(args,s) 
        records['number_val_samples'] = 100
        records['number_test_samples'] = len(dataset) // 32
        # Test metrics
        records['mean_test_loss'] = mean_test_loss
        records['test_acc'] = test_acc
        records['test_f1'] = test_f1
        records['test_prec'] = test_prec
        records['test_reca'] = test_reca
        records['test_roc'] = test_roc
        # Train metrics
        records['mean_train_loss'] = mean_train_loss
        records['train_acc'] = train_acc
        records['train_f1'] = train_f1
        records['train_prec'] = train_prec
        records['train_reca'] = train_reca
        records['train_roc'] = train_roc
        # Val metrics
        records['mean_val_loss'] = mean_val_loss
        records['val_acc'] = val_acc
        records['val_f1'] = val_f1
        records['val_prec'] = val_prec
        records['val_reca'] = val_reca
        records['val_roc'] = val_roc
        # Training metrics
        records['pytorch_total_params'] = sum(p.numel() for p in model.parameters())
        records['train_time'] = train_time
        records['best_epoch'] = best_epoch
        records['train_samples'] = '\n'.join(['{idx}/{samples_per_participant} v{video} {second_f}-{second_t}'.format_map(ts) for ts in train_samples])

        # Write results to csv
        with open(f'../logs/{args.test_results_dir}.csv','a') as fd:
            writer = csv.writer(fd)
            if os.stat(f'../logs/{args.test_results_dir}.csv'). st_size == 0:
                writer.writerow(records.keys())
            writer.writerow(records.items())