import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F

# Inputs to all functions are tensors

def accuracy_metric(outputs, targets):
    pred = torch.argmax(outputs,-1)
    return torch.sum(pred==targets).item() / targets.shape[0]

def f1_metric(outputs, targets):
    pred = torch.argmax(outputs,-1)
    return f1_score(targets.cpu().numpy(), pred.cpu().numpy(), average='macro', zero_division=0)

def precision_metric(outputs, targets):
    pred = torch.argmax(outputs,-1)
    return precision_score(targets.cpu().numpy(), pred.cpu().numpy(), average='macro', zero_division=0)

def recall_metric(outputs, targets):
    pred = torch.argmax(outputs,-1)
    return recall_score(targets.cpu().numpy(), pred.cpu().numpy(), average='macro', zero_division=0)

def roc_metric(outputs, targets, num_classes = 32):
    one_hot_targets = F.one_hot(targets, num_classes=num_classes)
    # print(torch.unique(targets))
    return roc_auc_score(one_hot_targets.cpu().numpy(), outputs.detach().cpu().numpy())