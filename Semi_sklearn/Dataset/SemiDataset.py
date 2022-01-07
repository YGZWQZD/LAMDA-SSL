from torch.utils.data import Dataset
from .SemiTrainDataset import SemiTrainDataset
from .LabledDataset import LabledDataset
from .UnlabledDataset import UnlabledDataset
from ..utils import get_indexing_method,get_len,indexing
from ..Split import SemiSplit
import torch
from scipy import sparse
class SemiDataset(Dataset):
    def __init__(self,
                test_size=None,
                labled_size=0.1,
                stratified=False,
                shuffle=True,
                random_state=None):
        super().__init__()
        self.labled_size=labled_size
        self.stratified=stratified
        self.shuffle=shuffle
        self.random_state=random_state
        self.test_size=test_size
        self.labled_X=None
        self.labled_y=None
        self.unlabled_X=None
        self.unlabled_y=None
        self.test_X=None
        self.test_y=None
        self.labled_dataset=None
        self.unlabled_dataset=None
        self.train_dataset=None
        self.test_dataset=None
        self.data_initialized=False
        self.len_test=None
        self.len_labled=None
        self.len_unlabled=None
        self.labled_X_indexing_method=None
        self.labled_y_indexing_method =None
        self.unlabled_X_indexing_method =None
        self.unlabled_y_indexing_method =None
        self.test_X_indexing_method=None
        self.test_y_indexing_method=None
        self.labled_class=LabledDataset
        self.unLabled_class=UnlabledDataset
        self.semitrain_class=SemiTrainDataset
    def _init_dataset(self):
        raise NotImplementedError(
            "_init_dataset method of SemiDataset class must be implemented."
        )

    def init_dataset(self,labled_X=None,labled_y=None,unlabled_X=None,
                    unlabled_y=None,test_X=None,test_y=None,train_dataset=None,
                    test_dataset=None,labled_dataset=None,unlabled_dataset=None):

        if labled_X is None and labled_dataset is None and train_dataset is None:
            self._init_dataset()

        if test_dataset is not None:
            self.test_dataset=test_dataset
        elif test_X is not None:
            self.test_dataset=self.labled_class()
            self.test_dataset.inin_dataset(test_X,test_y)
        elif self.test_size is not None:
            if labled_dataset is not None:
                self.test_dataset,labled_dataset=SemiSplit(dataset=labled_dataset,
                                                       labled_size=self.test_size,
                                                       stratified=self.stratified,
                                                       shuffle=self.shuffle,
                                                       random_state=self.random_state
                                                   )
            elif labled_X is not None:
                labled_X, labled_y, test_X, test_y = SemiSplit(X=labled_X, y=labled_y,
                                                           labled_size=self.test_size,
                                                           stratified=self.stratified,
                                                           shuffle=self.shuffle,
                                                           random_state=self.random_state
                                                    )
                self.test_dataset = self.labled_class()
                self.test_dataset.inin_dataset(self.test_X, self.test_y)

        if train_dataset is not None:
            self.train_dataset=train_dataset
        else:
            self.train_dataset=self.semitrain_class()
            self.train_dataset.init_dataset(labled_X=labled_X,labled_y=labled_y,unlabled_X=unlabled_X,
                    unlabled_y=unlabled_y,labled_dataset=labled_dataset,unlabled_dataset=unlabled_dataset)


        self.labled_dataset=self.train_dataset.get_dataset(labled=True)
        self.unlabled_dataset = self.train_dataset.get_dataset(labled=False)
        self.test_X = self.test_dataset.get_X()
        self.test_y = self.test_dataset.get_y()
        self.labled_X=self.labled_dataset.get_X()
        self.labled_y = self.labled_dataset.get_y()
        self.unlabled_X = self.unlabled_dataset.get_X()
        self.unlabled_y = self.unlabled_dataset.get_y()
        self.labled_X_indexing_method=get_indexing_method(self.labled_X)
        self.labled_y_indexing_method = get_indexing_method(self.labled_y)
        self.unlabled_X_indexing_method =get_indexing_method(self.unlabled_X)
        self.unlabled_y_indexing_method = get_indexing_method(self.unlabled_y)
        self.test_X_indexing_method = get_indexing_method(self.test_X)
        self.test_y_indexing_method = get_indexing_method(self.test_y)
        self.len_labled=self.labled_dataset.__len__()
        self.len_unlabled = self.unlabled_dataset.__len__()
        self.len_test = self.test_dataset.__len__()
        self.data_initialized=True

    def get_dataset(self,train=True,labled=None):
        if train:
            if labled is None:
                return self.train_dataset
            elif labled is True:
                return self.labled_dataset
            else:
                return self.unlabled_dataset
        else:
            return self.test_dataset

    def _transform(self,X,y):
        y = torch.Tensor([0]) if y is None else y
        if sparse.issparse(X):
            X = X.toarray().squeeze(0)
        return X, y

    def __getitem__(self, i, train=False, labled=True):
        if train is not True:
            X,y=self.test_X, self.test_y
            if hasattr(X, 'iloc'):
                X = {k: X[k].values.reshape(-1, 1) for k in X}
            Xi = indexing(X, i,self.test_X_indexing_method)
            yi = indexing(y, i,self.test_y_indexing_method)
        elif labled:
            X, y = self.labled_X, self.labled_y
            if hasattr(X, 'iloc'):
                X = {k: X[k].values.reshape(-1, 1) for k in X}
            Xi = indexing(X, i,self.labled_X_indexing_method)
            yi = indexing(y, i,self.labled_y_indexing_method)
        else:
            X,y= self.unlabled_X,self.unlabled_y
            if hasattr(X, 'iloc'):
                X = {k: X[k].values.reshape(-1, 1) for k in X}
            Xi = indexing(X, i,self.unlabled_X_indexing_method)
            yi = indexing(y, i, self.unlabled_y_indexing_method)
        return self._transform(Xi, yi)

    def __len__(self,train=False,labled=True):
        if train is not True:
            return self.len_test
        elif labled:
            return self.len_labled
        else:
            return self.len_unlabled





