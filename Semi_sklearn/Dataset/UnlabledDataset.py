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
    
class UblabledDataset(Dataset):
    def __init__(
            self,
            X,
            length=None,
    ):
        self.X = X
        self.X_indexing = check_indexing(X)
        self.X_is_ndframe = is_pandas_ndframe(X)
        if length is not None:
            self._len = length
            return
        # pylint: disable=invalid-name
        len_X = get_len(X)
        self._len = len_X

    def __len__(self):
        return self._len

    def transform(self, X):

        if sparse.issparse(X):
            X = X.toarray().squeeze(0)
        return X

    def __getitem__(self, i):
        X = self.X
        if self.X_is_ndframe:
            X = {k: X[k].values.reshape(-1, 1) for k in X}

        Xi = multi_indexing(X, i, self.X_indexing)
        return self.transform(Xi)