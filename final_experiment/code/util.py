import numpy as np
import math
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

def get_split_indices(target, number_train_samples, dataset_len, dont_shuffle_data = False):
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
            # Shuffle along first axis 
            np.random.shuffle(indices)
            # Shuffle along second axis 
            [np.random.shuffle(x) for x in indices]

        indices = indices.flatten()
        train_idx, val_idx = number_train_samples*32, number_train_samples*32 + 100 * 32

        train_mask = indices[:train_idx]
        val_mask = indices[train_idx:val_idx]
        test_mask = indices[val_idx:]
    else:
        raise 'Not implemented'
    return list(train_mask), list(val_mask), list(test_mask)