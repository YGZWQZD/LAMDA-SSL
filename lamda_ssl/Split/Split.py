import numpy as np
from sklearn.utils import _approximate_mode
from lamda_ssl.utils import get_len
from math import ceil
from sklearn.utils import check_random_state
from lamda_ssl.utils import to_numpy,get_indexing_method,indexing

def get_split_num(X,labeled_size=0.1):
    # print(labeled_size)
    len_X = get_len(X)
    labeled_size_type = np.asarray(labeled_size).dtype.kind
    # if labeled_size is not None and labeled_size_type not in ("i", "f"):
    #     raise ValueError("Invalid value for labeled_size: {}".format(labeled_size))
    if (
        labeled_size_type == "i"
        and (labeled_size >= len_X or labeled_size <= 0)
        or labeled_size_type == "f"
        and (labeled_size <= 0 or labeled_size >= 1)
        ):
        raise ValueError(
            "labeled_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(labeled_size, len_X)
        )

    if labeled_size_type == "f":
        num_labeled = ceil(labeled_size * len_X)
    else:

        num_labeled = labeled_size
    num_unlabeled=len_X-num_labeled
    return num_labeled,num_unlabeled

def get_split_index(y,num_labeled,num_unlabeled,stratified,shuffle,random_state=None):
    rng=check_random_state(seed=random_state)
    # print(num_labeled,num_unlabeled)
    num_total=num_labeled+num_unlabeled
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

        if num_unlabeled < num_classes :
            raise ValueError(
                "The unlabeled_size = %d should be greater or "
                "equal to the number of classes = %d" % (num_unlabeled, num_classes)
            )
        if num_labeled < num_classes :
            raise ValueError(
                "The labeled_size = %d should be greater or "
                "equal to the number of classes = %d" % (num_labeled, num_classes)
            )

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )
        n_i = _approximate_mode(class_counts, num_labeled, rng)
        class_counts_remaining = class_counts - n_i
        t_i = _approximate_mode(class_counts_remaining, num_unlabeled,rng)

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
            # print(num_total)
            permutation = rng.permutation(num_total)
        else:
            permutation = np.arange(num_total)
        ind_labeled = permutation[:num_labeled]
        ind_unlabeled = permutation[num_labeled : (num_labeled + num_unlabeled)]
    return ind_labeled,ind_unlabeled

def SemiSplit(stratified,shuffle,random_state=None, X=None, y=None,labeled_size=None):
        # print(labeled_size)
        num_labeled, num_unlabeled = get_split_num(X, labeled_size)
        ind_labeled, ind_unlabeled = get_split_index(y=y, num_labeled=num_labeled, num_unlabeled=num_unlabeled,
                                                   stratified=stratified, shuffle=shuffle,
                                                   random_state=random_state
                                                   )
        X_indexing = get_indexing_method(X)
        y_indexing = get_indexing_method(y)
        labeled_X = indexing(X, ind_labeled, X_indexing)
        labeled_y = indexing(y, ind_labeled, y_indexing)
        unlabeled_X = indexing(X, ind_unlabeled, X_indexing)
        unlabeled_y = indexing(y, ind_unlabeled, y_indexing)
        return labeled_X, labeled_y, unlabeled_X, unlabeled_y


        

