import numpy as np





def get_split_indices(target, number_targets, dataset_len, kfold_validation_offset = 0):

    n_samples = dataset_len // 32
    # Use number_test_targets videos from all participants as testing data
    if  target == 'participant_id':
        test_video_indices = np.arange(kfold_validation_offset,kfold_validation_offset + number_targets)                                                             
        test_mask = test_video_indices.copy()
        for i in range(1,32):
            # Wrap indices to allow for k fold cv offset
            test_video_indices = np.array([t if t < i*n_samples else t-n_samples for t in test_video_indices])
            test_video_indices += dataset_len//32
            test_mask = np.concatenate([test_mask,test_video_indices])
        train_mask = np.delete(np.arange(0,dataset_len),test_mask)
    elif target == 'video_id':
        raise 'test for sample size > 1'
        test_mask = np.arange(kfold_validation_offset*40, kfold_validation_offset*40 + number_targets*40)
        test_mask = np.array([t if t < dataset_len else t-dataset_len for t in test_mask])
        train_mask = np.delete(np.arange(0,dataset_len),test_mask)
    else:
        raise 'test for sample size > 1'
        # Subject-dependant emotion classification
        test_mask = np.arange(kfold_validation_offset,kfold_validation_offset + number_targets)
        # Wrap indices
        test_mask = np.array([t if t<dataset_len else t-dataset_len for t in test_mask])
        train_mask = np.delete(np.arange(0,dataset_len),test_mask)
    return list(train_mask), list(test_mask)