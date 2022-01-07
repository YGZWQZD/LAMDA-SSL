from torch.utils.data import Dataset
from scipy import sparse
import torch
from ..utils import get_indexing_method,get_len,indexing

    
class UnlabledDataset(Dataset):
    def __init__(
            self
    ):
        self.X=None
        self.y=None
        self.len=None
        self.data_initialized = False
        self.has_lable=False

    def init_dataset(self,X=None,y=None):
        self.X=X
        self.y=y

        self.len=get_len(self.X)
        self.X_indexing_method = get_indexing_method(self.X)
        self.y_indexing_method = get_indexing_method(self.y)
        if self.y is not None:
            self.has_lable=True

        self.data_initialized = True
    def __len__(self):
        return self.len

    def get_X(self):
        return self.X

    def get_y(self):
        if self.has_lable is not True:
            raise RuntimeError('No lables')
        return self.y

    def _transform(self,X,y):
        y = torch.Tensor([0]) if y is None else y
        if sparse.issparse(X):
            X = X.toarray().squeeze(0)
        return X, y

    def __getitem__(self, i):


        X=self.X
        y=self.y
        if hasattr(X, 'iloc'):
            X = {k: X[k].values.reshape(-1, 1) for k in X}

        Xi = indexing(X, i, self.X_indexing_method)
        yi = indexing(y, i, self.y_indexing_method)

        return self._transform(Xi,yi)
