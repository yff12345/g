import numpy as np

def get_split_indices(target, number_targets, dataset_len = 32*40):
    # Use number_test_targets videos from all participants as testing data
    if  target == 'participant_id':
        test_video_indices = np.arange(0,number_targets)
        test_mask = test_video_indices.copy()
        for i in range(1,32):
            test_video_indices += dataset_len//32
            test_mask = np.concatenate([test_mask,test_video_indices])
        train_mask = np.delete(np.arange(0,dataset_len),test_mask)
    elif target == 'video_id':
        raise 'err, recheck implementation'
        # train_mask = np.arange(0,number_targets*40)
        # test_mask = np.arange(number_targets*40,32*40-already_removed*32)
    else:
        # Subject-dependant emotion classification
        test_mask = np.arange(0,number_targets)
        train_mask = np.arange(number_targets, dataset_len)
    return list(train_mask), list(test_mask)