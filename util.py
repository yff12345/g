import numpy as np
import itertools
import torch

# Get 30 videos for each participant for training, 5 for validation and 5 for testing
# def train_val_test_split(dataset):
#   train_mask = np.append(np.repeat(1,30),np.repeat(0,10))
#   train_mask = np.tile(train_mask,int(len(dataset)/40))
#   val_mask = np.append(np.append(np.repeat(0,30),np.repeat(1,5)),np.repeat(0,5))
#   val_mask = np.tile(val_mask,int(len(dataset)/40))
#   test_mask = np.append(np.repeat(0,35),np.repeat(1,5))
#   test_mask = np.tile(test_mask,int(len(dataset)/40))

#   train_set = [c for c in itertools.compress(dataset,train_mask)]
#   val_set = [c for c in itertools.compress(dataset,val_mask)]
#   test_set = [c for c in itertools.compress(dataset,test_mask)]

#   return train_set, val_set, test_set

# Get x training videos for each participant for training and 40-x for testing
def train_test_split(dataset,x):
    assert x>=0 and x<=40

    train_mask = np.append(np.repeat(1,x),np.repeat(0,40-x))
    train_mask = np.tile(train_mask,int(len(dataset)/40))
    test_mask = np.append(np.repeat(0,x),np.repeat(1,40-x))
    test_mask = np.tile(test_mask,int(len(dataset)/40))

    train_set = [c for c in itertools.compress(dataset,train_mask)]
    test_set = [c for c in itertools.compress(dataset,test_mask)]

    return train_set, test_set


def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    # f1.requires_grad = is_training
    return f1