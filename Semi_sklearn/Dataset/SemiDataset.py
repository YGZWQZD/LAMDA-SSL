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

class SemiDataset(Dataset):
    def __init__(
            self,
            labled_X,
            labled_y=None,
            unlabled_X=None,
            labled_length=None,
            unlabled_length=None
    ):
        self.labled_X = labled_X
        self.labled_X_indexing = check_indexing(labled_X)
        self.lableed_X_is_ndframe = is_pandas_ndframe(labled_X)
        self.unlabled_X = unlabled_X
        self.unlabled_X_indexing = check_indexing(unlabled_X)
        self.unlabled_X_is_ndframe = is_pandas_ndframe(unlabled_X)
        if labled_length is not None:
            self._labled_len = labled_length
        else:
            labled_len_X = get_len(labled_X)
            self._labled_len = labled_len_X
        if unlabled_length is not None:
            self._unlabled_len = unlabled_length
        else:
            unlabled_len_X = get_len(labled_X)
            self._unlabled_len = unlabled_len_X

    def __len__(self,labled = True):
        if labled:
            return self._labled_len
        else:
            return self._unlabled_len

    def transform(self, X, y, labled = True):
        if labled:
            y = torch.Tensor([0]) if y is None else y

            if sparse.issparse(X):
                X = X.toarray().squeeze(0)
            return X, y
        else:
            if sparse.issparse(X):
                X = X.toarray().squeeze(0)
            return X

    def __getitem__(self, i, labled = True):
        if labled:
            X, y = self.X, self.y
            if self.X_is_ndframe:
                X = {k: X[k].values.reshape(-1, 1) for k in X}

            Xi = multi_indexing(X, i, self.X_indexing)
            yi = multi_indexing(y, i, self.y_indexing)
            return self.transform(Xi, yi)
        else:
            X = self.X
            if self.X_is_ndframe:
                X = {k: X[k].values.reshape(-1, 1) for k in X}

            Xi = multi_indexing(X, i, self.X_indexing)
            return self.transform(Xi)