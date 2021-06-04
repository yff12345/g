
import torch
import csv   
import numpy as np
from torch_geometric.data import  DataLoader
from metrics import accuracy_metric, f1_metric, precision_metric, recall_metric, roc_metric

def test_epoch(model, loader ,criterion,args):
    model.eval()
    losses, outputs, targets = [], [] , []
    for batch in loader:
        batch = batch.to(args.device)
        y = batch.y if args.target != 'emotion_labels' else (batch.y[:,args.target_emotion] > 5).long()
        out = model(batch)
        outputs.append(out)
        targets.append(y)
        loss = criterion(out,y)
        losses.append(loss.item())

    print(torch.cat(outputs), torch.cat(targets))
    return np.array(losses).mean(),torch.cat(outputs), torch.cat(targets)


def main(model, dataset, criterion , args):
    model.load_state_dict(torch.load(f'./{args.test_model_dict}'))
    test_loader = DataLoader(dataset, batch_size=args.batch_size)
    # Test epoch
    mean_test_loss, test_outputs, test_targets = test_epoch(model, test_loader ,criterion,args)
    # Metrics
    test_acc = accuracy_metric(test_outputs, test_targets)
    test_f1 = f1_metric(test_outputs, test_targets)
    test_prec = precision_metric(test_outputs, test_targets)
    test_reca = recall_metric(test_outputs, test_targets)
    # test_roc = roc_metric(test_outputs, test_targets)
    test_roc = -1
    
    print(f'\n--Testing--')
    print(f'Test loss: {mean_test_loss:.5f}')
    print(f'Test acc: {test_acc:.3f}  ')
    print(f'Test F1: {test_f1:.3f} ')
    print(f'Test precision: {test_prec:.3f}')
    print(f'Test recall: {test_reca:.3f}')
    # print(f'Test ROC auc: {test_roc:.3f} \n')


    if args.write_test_results:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        # Write results to csv
        # fields = "model - eeg_feature - remove_global_connections - remove_baseline_signal_noise_removal - number_test_targets - batch_size - hidden_channels - learning_rate - dropout_rate - mean_test_loss - test_acc - test_f1 - test_prec - test_reca - test_roc - pytorch_total_params"
        # fields = fields.replace(" ",'').split('-')
        # print(fields)
        # fields=['first','second','third']
        fields = [args.model, args.eeg_feature, args.remove_global_connections, args.remove_baseline_signal_noise_removal, args.number_test_targets, args.batch_size, args.hidden_channels, args.learning_rate,args.dropout_rate, mean_test_loss, test_acc, test_f1, test_prec, test_reca, test_roc, pytorch_total_params ]
        with open(f'{args.target}_{args.test_results_dir}.csv','a') as fd:
            writer = csv.writer(fd)
            writer.writerow(fields)