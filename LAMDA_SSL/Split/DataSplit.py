import numpy as np
from sklearn.utils import _approximate_mode
from LAMDA_SSL.utils import get_len
from math import ceil
from sklearn.utils import check_random_state
from LAMDA_SSL.utils import to_numpy,get_indexing_method,indexing

def get_split_num(X,size_split=0.1):
    len_X = get_len(X)
    type_size_split = np.asarray(size_split).dtype.kind
    if (
        type_size_split == "i"
        and (size_split >= len_X or size_split <= 0)
        or type_size_split == "f"
        and (size_split <= 0 or size_split >= 1)
        ):
        raise ValueError(
            "size_split={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(size_split, len_X)
        )

    if type_size_split == "f":
        num_1 = ceil(size_split * len_X)
    else:

        num_1= size_split
    num_2=len_X-num_1
    return num_1,num_2

def get_split_index(y,num_1,num_2,stratified,shuffle,random_state=None):
    rng=check_random_state(seed=random_state)
    num_total=num_1+num_2
    if stratified:
        try:
            y_arr=to_numpy(y)
        except (AttributeError, TypeError):
            y_arr = y
        if y_arr.ndim == 2:
            y_arr = np.array([" ".join(row.astype("str")) for row in y_arr])
        classes, y_indices = np.unique(y_arr, return_inverse=True)
        num_classes = classes.shape[0]
        class_counts = np.bincount(y_indices)
        if np.min(class_counts) < 2:
            raise ValueError(
                "The least populated class in y has only 1"
                " member, which is too few. The minimum"
                " number of groups for any class cannot"
                " be less than 2."
            )

        if num_1 < num_classes :
            raise ValueError(
                "The num_1 = %d should be greater or "
                "equal to the number of classes = %d" % (num_1, num_classes)
            )
        if num_2< num_classes :
            raise ValueError(
                "The num_2 = %d should be greater or "
                "equal to the number of classes = %d" % (num_2, num_classes)
            )

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )
        n_i = _approximate_mode(class_counts, num_1, rng)
        class_counts_remaining = class_counts - n_i
        t_i = _approximate_mode(class_counts_remaining, num_2,rng)

        ind_unlabeled = []
        ind_labeled = []

        for i in range(num_classes):
            if shuffle:
                permutation = rng.permutation(class_counts[i])
            else:
                permutation = np.arange(class_counts[i])
            perm_indices_class_i = class_indices[i].take(permutation, mode="clip")
            ind_labeled.extend(perm_indices_class_i[: n_i[i]])
            ind_unlabeled.extend(perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]])
        if shuffle:
            ind_labeled = rng.permutation(ind_labeled)
            ind_unlabeled = rng.permutation(ind_unlabeled)
    else:
        if shuffle:
            permutation = rng.permutation(num_total)
        else:
            permutation = np.arange(num_total)
        ind_labeled = permutation[:num_1]
        ind_unlabeled = permutation[num_1 : (num_1+ num_2)]
    return ind_labeled,ind_unlabeled

def DataSplit(stratified=True,shuffle=True,random_state=None, X=None, y=None,size_split=None):
    # >> Parameter
    # >> - stratified: Whether to stratify by classes.
    # >> - shuffle: Whether to shuffle the data.
    # >> - random_state: The random seed.
    # >> - X: Samples of the data to be split.
    # >> - y: Labels of the data to be split.
    # >> - labeled_size: The scale or size of the labeled data.
    num_1, num_2 = get_split_num(X, size_split)
    ind_1, ind_2 = get_split_index(y=y, num_1=num_1, num_2=num_2,
                                   stratified=stratified, shuffle=shuffle,
                                   random_state=random_state)
    X_indexing = get_indexing_method(X)
    y_indexing = get_indexing_method(y)
    X_1 = indexing(X, ind_1, X_indexing)
    y_1 = indexing(y, ind_1, y_indexing)
    X_2 = indexing(X, ind_2, X_indexing)
    y_2 = indexing(y, ind_2, y_indexing)
    return X_1, y_1, X_2, y_2


        

