from torch.utils.data import Dataset
from ..utils import get_indexing_method,get_len,indexing
from .LabledDataset import LabledDataset
from .UnlabledDataset import UnlabledDataset
from ..Split.SemiSplit import SemiSplit
import torch
from scipy import sparse
class SemiTrainDataset(Dataset):
    def __init__(self,
                labled_size=0.1,
                stratified=False,
                shuffle=True,
                random_state=None):
        super().__init__()
        self.labled_size=labled_size
        self.stratified=stratified
        self.shuffle=shuffle
        self.random_state=random_state
        self.labled_X=None
        self.labled_y=None
        self.unlabled_X=None
        self.unlabled_y=None
        self.labled_dataset=None
        self.unlabled_dataset=None
        self.data_initialized=False
        self.len_labled=None
        self.len_unlabled=None
        self.labled_class=LabledDataset
        self.unLabled_class=UnlabledDataset
    def _init_dataset(self):
        raise NotImplementedError(
            "_init_dataset method of SemiTrainDataset class must be implemented."
        )

    def init_dataset(self,labled_X=None,labled_y=None,unlabled_X=None,
                    unlabled_y=None,labled_dataset=None,unlabled_dataset=None):
        if labled_X is not None:
            if unlabled_X is not None:
                self.unlabled_X = unlabled_X
                self.unlabled_y = unlabled_y
                self.labled_X=labled_X
                self.labled_y=labled_y
            else:
                self.labled_X,self.labled_y,self.unlabled_X,self.unlabled_y=SemiSplit(X=labled_X,y=labled_y,
                                                                labled_size=self.labled_size,
                                                                stratified=self.stratified,
                                                                shuffle=self.shuffle,
                                                                random_state=self.random_state
                                                                )
            self.unlabled_dataset = UnlabledDataset()
            self.unlabled_dataset.init_dataset(self.unlabled_X, self.unlabled_y)
            self.labled_dataset=LabledDataset()
            self.labled_dataset.init_dataset(self.labled_X,self.labled_y)

        elif labled_dataset is not None:
            self.labled_dataset=labled_dataset
            if unlabled_X is not None:
                self.unlabled_dataset=unlabled_dataset
            else:
                self.labled_dataset,self.unlabled_dataset=SemiSplit(dataset=self.labled_dataset,
                                                                labled_size=self.labled_size,
                                                                stratified=self.stratified,
                                                                shuffle=self.shuffle,
                                                                random_state=self.random_state
                                                                )
            self.labled_X = self.labled_dataset.get_X()
            self.labled_y = self.labled_dataset.get_y()
            self.unlabled_X=self.unlabled_dataset.get_X()
            self.unlabled_y = self.unlabled_dataset.get_y()
        else:
            self._init_dataset()
        self.labled_X_indexing_method=get_indexing_method(self.labled_X)
        self.labled_y_indexing_method = get_indexing_method(self.labled_y)
        self.unlabled_X_indexing_method=get_indexing_method(self.unlabled_X)
        self.unlabled_y_indexing_method = get_indexing_method(self.unlabled_y)
        self.len_labled=self.labled_dataset.__len__()
        self.len_unlabled = self.unlabled_dataset.__len__()

        self.data_initialized=True
    def get_dataset(self,labled):
        if labled:
            return self.labled_dataset
        else:
            return self.unlabled_dataset
    def _transform(self,X,y):
        y = torch.Tensor([0]) if y is None else y
        if sparse.issparse(X):
            X = X.toarray().squeeze(0)
        return X, y
    def __getitem__(self, i, labled=True):
        if labled:
            X, y = self.labled_X, self.labled_y
            if hasattr(X, 'iloc'):
                X = {k: X[k].values.reshape(-1, 1) for k in X}
            Xi = indexing(X, i,self.labled_X_indexing_method)
            yi = indexing(y, i,self.labled_y_indexing_method)
        else:
            X, y = self.unlabled_X, self.unlabled_y
            if hasattr(X, 'iloc'):
                X = {k: X[k].values.reshape(-1, 1) for k in X}
            Xi = indexing(X, i,self.unlabled_X_indexing_method)
            yi = indexing(y, i,self.unlabled_X_indexing_method)
        return self._transform(Xi, yi)
    def __len__(self,labled=True):
        if labled:
            return self.len_labled
        else:
            return self.len_unlabled





