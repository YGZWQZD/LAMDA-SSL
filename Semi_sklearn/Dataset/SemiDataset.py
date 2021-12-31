from torch.utils.data import Dataset
from scipy import sparse
import numpy as np
import torch
from skorch.utils import flatten
from skorch.utils import is_pandas_ndframe
from skorch.utils import check_indexing
from skorch.utils import multi_indexing
from skorch.utils import to_numpy
from sklearn.utils import check_random_state
from Semi_sklearn.Split.SemiSplit import SemiSplit
from skorch.dataset import get_len


class SemiDataset(Dataset):
    def __init__(
            self,
            labled_X,
            labled_y=None,
            unlabled_X=None,
            labled_size=0.1,
            stratified=False,
            shuffle=True,
            random_state=None
            ):
        len_labled_X = get_len(labled_X)
        if labled_y is not None:
            len_labled_y = get_len(labled_y)
            if len_labled_y != len_labled_X:
                raise ValueError("labled_X and labled_y have inconsistent lengths.")
        self.labled_X_indexing = check_indexing(labled_X)
        self.labled_y_indexing = check_indexing(labled_y)
        self.labled_X_is_ndframe = is_pandas_ndframe(labled_X)
        if unlabled_X is not None:
            self.unlabled_X = unlabled_X
            self.unlabled_y = None
            self.labled_y = labled_y
            self.labled_X = labled_X
        else:
            self.labled_X,self.labled_y,self.unlabled_X,self.unlabled_y=SemiSplit(X=labled_X,y=labled_y,
                                                                labled_size=labled_size,
                                                                stratified=stratified,
                                                                shuffle=shuffle,
                                                                random_state=random_state,
                                                                X_indexing=self.labled_X_indexing, 
                                                                y_indexing=self.labled_y_indexing
                                                                )

        self.unlabled_X_indexing = check_indexing(self.unlabled_X)
        self.unlabled_y_indexing = check_indexing(self.unlabled_y)
        self.unlabled_X_is_ndframe = is_pandas_ndframe(self.unlabled_X)
        len_labled_X = get_len(self.labled_X)
        len_labled_y = get_len(self.labled_y)
        self._labled_len = len_labled_X
        unlabled_len_X = get_len(self.unlabled_X)
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
            X, y = self.labled_X, self.labled_y
            if self.labled_X_is_ndframe:
                X = {k: X[k].values.reshape(-1, 1) for k in X}

            Xi = multi_indexing(X, i, self.labled_X_indexing)
            yi = multi_indexing(y, i, self.labled_y_indexing)
            return self.transform(Xi, yi, labled)
        else:
            X = self.unlabled_X
            if self.unlabled_X_is_ndframe:
                X = {k: X[k].values.reshape(-1, 1) for k in X}

            Xi = multi_indexing(X, i, self.unlabled_X_indexing)
            return self.transform(Xi, labled)

    def getitem_unlabled_y(self,i,labled=False):
        # For evaluation of transductive learning
        y=self.unlabled_y
        if y is None:
            return y
        yi = multi_indexing(y, i, self.unlabled_y_indexing)
        yi = torch.Tensor([0]) if y is None else y
        return yi