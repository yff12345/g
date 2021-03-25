import numpy as np
import itertools

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