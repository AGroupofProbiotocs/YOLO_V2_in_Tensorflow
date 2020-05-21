import numpy as np

def cal_feat_size(input_size, num_pooling=5, kernel_size=2, padding='same', network='darknet448'):
    '''
    Calculate the feature size according to the input size, used only for the network using pooling to downsample.
    '''

    feat_size = np.array(input_size)

    if network == 'darknet448':
        return feat_size / 32
    else:
        pass

    if padding == 'same':
        int_func = np.floor
    elif padding == 'valid':
        int_func = np.ceil
    else:
        raise ValueError("padding should be 'same' or 'valid'!")

    for i in range(num_pooling):
        feat_size = int_func(feat_size / kernel_size)

    return feat_size