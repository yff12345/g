import numpy as np

def get_split_indices(target, number_targets, dataset_len = 32*40, kfold_validation_offset = 0):
    # Use number_test_targets videos from all participants as testing data
    if  target == 'participant_id':
        test_video_indices = np.arange(kfold_validation_offset,kfold_validation_offset + number_targets)
        test_mask = test_video_indices.copy()
        for i in range(1,32):
            test_video_indices += dataset_len//32
            test_mask = np.concatenate([test_mask,test_video_indices])
        train_mask = np.delete(np.arange(0,dataset_len),test_mask)
    elif target == 'video_id':
        test_mask = np.arange(kfold_validation_offset*40, kfold_validation_offset*40 + number_targets*40)
        test_mask = np.array([t if t < dataset_len else t-dataset_len for t in test_mask])
        train_mask = np.delete(np.arange(0,dataset_len),test_mask)
    else:
        # Subject-dependant emotion classification
        test_mask = np.arange(kfold_validation_offset,kfold_validation_offset + number_targets)
        # Wrap indices
        test_mask = np.array([t if t<dataset_len else t-dataset_len for t in test_mask])
        train_mask = np.delete(np.arange(0,dataset_len),test_mask)
    return list(train_mask), list(test_mask)