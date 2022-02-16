import numpy as np
from sklearn.utils.validation import check_array
from sklearn.utils import _approximate_mode
from Semi_sklearn.utils import get_len
from math import ceil
from sklearn.utils import check_random_state
from Semi_sklearn.utils import to_numpy,get_indexing_method,indexing

def get_split_num(X,labled_size=0.1):
    len_X = get_len(X)
    labled_size_type = np.asarray(labled_size).dtype.kind
    # if labled_size is not None and labled_size_type not in ("i", "f"):
    #     raise ValueError("Invalid value for labled_size: {}".format(labled_size))
    if (
        labled_size_type == "i"
        and (labled_size >= len_X or labled_size <= 0)
        or labled_size_type == "f"
        and (labled_size <= 0 or labled_size >= 1)
        ):
        raise ValueError(
            "labled_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(labled_size, len_X)
        )

    if labled_size_type == "f":
        num_labled = ceil(labled_size * len_X)
    else:
        num_labled = float(labled_size)
    num_unlabled=len_X-num_labled
    return num_labled,num_unlabled

def get_split_index(y,num_labled,num_unlabled,stratified,shuffle,random_state=None):
    rng=check_random_state(seed=random_state)
    num_total=num_labled+num_unlabled
    if stratified:
        try:
            y_arr=to_numpy(y)
        except (AttributeError, TypeError):
            y_arr = y
        if y_arr.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
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

        if num_unlabled < num_classes :
            raise ValueError(
                "The unlabled_size = %d should be greater or "
                "equal to the number of classes = %d" % (num_unlabled, num_classes)
            )
        if num_labled < num_classes :
            raise ValueError(
                "The labled_size = %d should be greater or "
                "equal to the number of classes = %d" % (num_labled, num_classes)
            )

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )
        n_i = _approximate_mode(class_counts, num_labled, rng)
        class_counts_remaining = class_counts - n_i
        t_i = _approximate_mode(class_counts_remaining, num_unlabled,rng)

        ind_unlabled = []
        ind_labled = []

        for i in range(num_classes):
            if shuffle:
                permutation = rng.permutation(class_counts[i])
            else:
                permutation = np.arange(class_counts[i])
            perm_indices_class_i = class_indices[i].take(permutation, mode="clip")
            ind_labled.extend(perm_indices_class_i[: n_i[i]])
            ind_unlabled.extend(perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]])
        if shuffle:
            ind_labled = rng.permutation(ind_labled)
            ind_unlabled = rng.permutation(ind_unlabled)
    else:
        if shuffle:
            permutation = rng.permutation(num_total)
        else:
            permutation = np.arange(num_total)
        ind_labled = permutation[:num_labled]
        ind_unlabled = permutation[num_labled : (num_labled + num_unlabled)]
    return ind_labled,ind_unlabled

def SemiSplit(stratified,shuffle,random_state=None, X=None, y=None,labled_size=None):
        num_labled, num_unlabled = get_split_num(X, labled_size)
        ind_labled, ind_unlabled = get_split_index(y=y, num_labled=num_labled, num_unlabled=num_unlabled,
                                                   stratified=stratified, shuffle=shuffle,
                                                   random_state=random_state
                                                   )
        X_indexing = get_indexing_method(X)
        y_indexing = get_indexing_method(y)
        labled_X = indexing(X, ind_labled, X_indexing)
        labled_y = indexing(y, ind_labled, y_indexing)
        unlabled_X = indexing(X, ind_unlabled, X_indexing)
        unlabled_y = indexing(y, ind_unlabled, y_indexing)
        return labled_X, labled_y, unlabled_X, unlabled_y


        

