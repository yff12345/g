import numpy as np
import math
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from einops import rearrange

def get_split_indices(target, number_train_samples, dataset_len, dont_shuffle_data = False,train_val_sample_array=[]):
    windows_per_participant = dataset_len // 32
    # Use number_train_samples as training samples, 100 for validation and the remaining for testing 
    # if number_train_samples + 200 > windows_per_participant:
    #     print('Error, number_train_samples is too big')
    #     exit()
    
    if  target == 'participant_id' or target == 'video_id':
        indices = np.arange(0,dataset_len).reshape((32,40,windows_per_participant // 40))
        # Define indices in shape p,v,w then transpose to w,v,p for participant classification and w,p,v for video classification
        indices = indices.transpose(2,1,0) if target == 'participant_id' else indices.transpose(2,0,1)

        if not dont_shuffle_data:
            # TODO: Shuffle
            print('Shuffling dataset before sub sampling train/val/test')
            # Shuffle along first and second axis 
            orig_shape = indices.shape
            indices = rearrange(indices, 'a b c -> (a b) c')
            np.random.shuffle(indices)
            indices.reshape(orig_shape)

        indices = indices.flatten()
        sample_size = 32 if target == 'participant_id' else 40
        if len(train_val_sample_array) > 0:
            # First 100 samples are validation, the remaining is training
            train_val_indices = np.array([np.arange(np.where(indices == idx)[0][0], np.where(indices == idx)[0][0]+32) for idx in train_val_sample_array]).flatten()
            train_idx, val_idx = train_val_indices[100*sample_size:], train_val_indices[:100*sample_size]
            train_mask = indices[train_idx]
            val_mask = indices[val_idx]
            test_mask = list(filter(lambda a: a not in train_mask and a not in val_mask, indices))

        else:
            train_idx, val_idx = number_train_samples*sample_size, number_train_samples*sample_size + 100 * sample_size
            train_mask = indices[:train_idx]
            val_mask = indices[train_idx:val_idx]
            test_mask = indices[val_idx:]
    else:
        raise 'Not implemented'
    return list(train_mask), list(val_mask), list(test_mask)