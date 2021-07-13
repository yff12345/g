import numpy as np
import math

def get_split_indices(target, train_split,val_split, dataset_len, window_size, kfold_validation_offset = 0):
    windows_per_participant = int(40*(60/window_size))
    windows_per_video = windows_per_participant // 40
    # Use number_test_targets videos from all participants as testing data
    if  target == 'participant_id' or target == 'video_id':
        indices = np.arange(0,dataset_len).reshape((32,40,windows_per_video))
        # Define indices in shape p,v,w then transpose to w,v,p for participant classification and w,p,v for video classification
        indices = indices.transpose(2,1,0).flatten() if target == 'participant_id' else indices.transpose(2,0,1).flatten() 
        if train_split == -1:
            val_n_samples = math.ceil(dataset_len*val_split*.01)
            train_mask = indices[:32 if target == 'participant_id' else 40]
            val_mask = indices[32:val_n_samples]
            test_mask = indices[val_n_samples:]
        else:
            train_n_samples = math.ceil(dataset_len*train_split*.01)
            val_n_samples = math.ceil(dataset_len*val_split*.01)
            train_mask = indices[:train_n_samples]
            val_mask = indices[train_n_samples:train_n_samples+val_n_samples]
            test_mask = indices[train_n_samples+val_n_samples:]
    else:
        raise 'test for sample size > 1'
        # Subject-dependant emotion classification
        test_mask = np.arange(kfold_validation_offset,kfold_validation_offset + number_targets)
        # Wrap indices
        test_mask = np.array([t if t<dataset_len else t-dataset_len for t in test_mask])
        train_mask = np.delete(np.arange(0,dataset_len),test_mask)
    return list(train_mask), list(val_mask), list(test_mask)