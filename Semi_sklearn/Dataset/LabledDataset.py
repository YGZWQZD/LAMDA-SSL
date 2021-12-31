from torch.utils.data import Dataset
from scipy import sparse
import numpy as np
import torch
from skorch.utils import flatten
from skorch.utils import is_pandas_ndframe
from skorch.utils import check_indexing
from skorch.utils import multi_indexing
from skorch.utils import to_numpy

def get_len(data):
    lens = [_apply_to_data(data, _len, unpack_dict=True)]
    lens = list(flatten(lens))
    len_set = set(lens)
    if len(len_set) != 1:
        raise ValueError("Dataset does not have consistent lengths.")
    return list(len_set)[0]


class LabledDataset(Dataset):
    def __init__(
            self,
            X,
            y=None,
            length=None,
    ):
        self.X = X
        self.y = y

        self.X_indexing = check_indexing(X)
        self.y_indexing = check_indexing(y)
        self.X_is_ndframe = is_pandas_ndframe(X)

        if length is not None:
            self._len = length
            return

        # pylint: disable=invalid-name
        len_X = get_len(X)
        if y is not None:
            len_y = get_len(y)
            if len_y != len_X:
                raise ValueError("X and y have inconsistent lengths.")
        self._len = len_X

    def __len__(self):
        return self._len

    def transform(self, X, y):
        y = torch.Tensor([0]) if y is None else y

        if sparse.issparse(X):
            X = X.toarray().squeeze(0)
        return X, y

    def __getitem__(self, i):
        X, y = self.X, self.y
        if self.X_is_ndframe:
            X = {k: X[k].values.reshape(-1, 1) for k in X}

        Xi = multi_indexing(X, i, self.X_indexing)
        yi = multi_indexing(y, i, self.y_indexing)
        return self.transform(Xi, yi)