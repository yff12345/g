import torch
from sklearn.metrics import precision_score, recall_score, f1_score

# Inputs to all functions are tensors

def accuracy_metric(outputs, targets):
    pred = torch.argmax(outputs,-1)
    return torch.sum(pred==targets).item() / targets.shape[0]

def f1_metric(outputs, targets):
    pred = torch.argmax(outputs,-1)
    return f1_score(targets.cpu().numpy(), pred.cpu().numpy(), average='micro')

def precision_metric(outputs, targets):
    pred = torch.argmax(outputs,-1)
    return precision_score(targets.cpu().numpy(), pred.cpu().numpy(), average='micro')

def recall_metric(outputs, targets):
    pred = torch.argmax(outputs,-1)
    return recall_score(targets.cpu().numpy(), pred.cpu().numpy(), average='micro')
