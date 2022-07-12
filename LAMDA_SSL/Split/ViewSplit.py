import copy

import numpy as np

def ViewSplit(X,num_splits=2,axis=1,shuffle=True):
    # >> Parameter
    # >> - X: Samples of the data to be split.
    # >> - num_splits: The number of views
    # >> - axis: The axis of the dimension to be splited.
    # >> - shuffle: Whether to shuffle the features.
    shape=X.shape
    range_shape=tuple(i for i in range(len(shape)))
    pre=range_shape[:axis]
    suf = range_shape[axis+1:]
    num_features=shape[axis]
    cur_array=np.arange(num_features)
    num_features_view=num_features//num_splits
    mod=num_features%num_splits
    idx=[]
    for _ in range(num_splits):
        if _ < mod:
            cur_size=num_features_view+1
        else:
            cur_size = num_features_view
        if shuffle is True:
            cur_idx = np.random.choice(cur_array, size=cur_size, replace=False)
        else:
            cur_idx= cur_array[:cur_size]
        cur_array = np.array([i for i in cur_array if i not in cur_idx])
        idx.append(cur_idx)
    _X=copy.copy(X)
    _X=_X.transpose(tuple([axis])+pre+suf)
    result = []
    for cur_idx in idx:
        cur_X=_X[cur_idx]
        cur_X=cur_X.transpose(range_shape[1:axis+1]+tuple([0])+range_shape[axis+1:])
        result.append(cur_X)
    return result



