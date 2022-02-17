from torch.utils.data import Dataset
from .TrainDataset import TrainDataset
from .LabledDataset import LabledDataset
from .UnlabledDataset import UnlabledDataset
from ..utils import get_indexing_method
from ..Split.SemiSplit import SemiSplit
# from Semi_sklearn.utils import partial
# import torch
# from scipy import sparse
class SemiDataset(Dataset):
    def __init__(self,
                 transforms=None,
                 transform=None,
                 target_transform=None,
                 unlabled_transform=None,
                 valid_transform=None,
                 test_transform=None,
                 test_size=None,
                 valid_size=None,
                 labled_size=None,
                 stratified=False,
                 shuffle=True,
                 random_state=None):
        self.transforms=transforms
        self.transform = transform
        self.target_transform=target_transform
        self.unlabled_transform = unlabled_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform

        self.labled_size=labled_size
        self.valid_size=valid_size
        self.test_size = test_size

        self.stratified=stratified
        self.shuffle=shuffle
        self.random_state=random_state

        self.labled_X=None
        self.labled_y=None
        self.unlabled_X=None
        self.unlabled_y=None
        self.valid_X=None
        self.valid_y=None
        self.test_X=None
        self.test_y=None

        self.labled_dataset=LabledDataset(transforms=self.transforms, transform=self.transform,
                                          target_transform=self.target_transform)
        self.unlabled_dataset=UnlabledDataset(transform=self.unlabled_transform)
        self.train_dataset = TrainDataset(transforms=self.transforms, transform=self.transform,
                                          target_transform=self.target_transform,
                                          unlabled_transform=self.unlabled_transform,
                                          labled_size=self.labled_size, stratified=self.stratified,
                                          shuffle=self.shuffle, random_state=self.random_state)
        self.valid_dataset = LabledDataset(transform=self.valid_transform)
        self.test_dataset=LabledDataset(transform=self.test_transform)

        self.len_test=None
        self.len_valid = None
        self.len_labled=None
        self.len_unlabled=None

        self.labled_X_indexing_method=None
        self.labled_y_indexing_method =None
        self.unlabled_X_indexing_method =None
        self.unlabled_y_indexing_method =None
        self.test_X_indexing_method=None
        self.test_y_indexing_method=None
        self.valid_X_indexing_method=None
        self.valid_y_indexing_method=None

        self.data_initialized=False
        # self.labled_class=LabledDataset
        # self.unLabled_class=UnlabledDataset
        # self.train_class=partial(TrainDataset,labled_size=self.labled_size,stratified=self.stratified,
        #                              shuffle=self.shuffle,random_state=self.random_state)

    def _init_dataset(self):
        raise NotImplementedError(
            "_init_dataset method of SemiDataset class must be implemented."
        )

    def init_dataset(self,labled_X=None,labled_y=None,unlabled_X=None,
                    unlabled_y=None,test_X=None,test_y=None,valid_X=None,valid_y=None,train_dataset=None,
                    test_dataset=None,valid_dataset=None,labled_dataset=None,unlabled_dataset=None):

        if labled_X is None and labled_dataset is None and train_dataset is None:
            self._init_dataset()
        elif train_dataset is not None:
            self.train_dataset = train_dataset
            if test_dataset is not None:
                self.test_dataset = test_dataset
            elif test_X is not None:
                self.test_dataset.inin_dataset(test_X, test_y)
            if valid_dataset is not None:
                self.valid_dataset = valid_dataset
            elif valid_X is not None:
                self.valid_dataset.inin_dataset(test_X, test_y)
        else:
            if labled_dataset is not None:
                labled_X = getattr(labled_dataset, 'X')
                labled_y = getattr(labled_dataset, 'y')

            if test_dataset is not None:
                self.test_dataset=test_dataset
                # setattr(self.test_dataset,'transform',self.test_transform)
            elif test_X is not None:
                self.test_dataset.inin_dataset(test_X,test_y)
            elif self.test_size is not None:
                test_X, test_y,labled_X, labled_y = SemiSplit(X=labled_X, y=labled_y,
                                                           labled_size=self.test_size,
                                                           stratified=self.stratified,
                                                           shuffle=self.shuffle,
                                                           random_state=self.random_state
                                                        )
                self.test_dataset.inin_dataset(self.test_X, self.test_y)

            if valid_dataset is not None:
                self.valid_dataset=valid_dataset
            elif valid_X is not None:
                self.valid_dataset.inin_dataset(valid_X,valid_y)
            elif self.valid_size is not None:
                if labled_dataset is not None:
                    labled_X=getattr(labled_dataset,'X')
                    labled_y=getattr(labled_dataset,'y')

                valid_X, valid_y,labled_X, labled_y = SemiSplit(X=labled_X, y=labled_y,
                                                           labled_size=self.valid_size,
                                                           stratified=self.stratified,
                                                           shuffle=self.shuffle,
                                                           random_state=self.random_state
                                                        )
                self.valid_dataset.inin_dataset(self.test_X, self.test_y)


            self.train_dataset.init_dataset(labled_X=labled_X,labled_y=labled_y,unlabled_X=unlabled_X,
                    unlabled_y=unlabled_y,labled_dataset=labled_dataset,unlabled_dataset=unlabled_dataset)

        self.labled_dataset=self.train_dataset.get_dataset(labled=True)
        self.unlabled_dataset=self.train_dataset.get_dataset(labled=False)

        self.test_X = getattr(self.test_dataset,'X')
        self.test_y = getattr(self.test_dataset,'y')
        self.valid_X = getattr(self.valid_dataset,'X')
        self.valid_y = getattr(self.valid_dataset,'y')
        self.labled_X=getattr(self.labled_dataset,'X')
        self.labled_y = getattr(self.labled_dataset,'y')
        self.unlabled_X = getattr(self.unlabled_dataset,'X')
        self.unlabled_y = getattr(self.unlabled_dataset,'y')

        self.labled_X_indexing_method=get_indexing_method(self.labled_X)
        self.labled_y_indexing_method = get_indexing_method(self.labled_y)
        self.unlabled_X_indexing_method =get_indexing_method(self.unlabled_X)
        self.unlabled_y_indexing_method = get_indexing_method(self.unlabled_y)
        self.valid_X_indexing_method = get_indexing_method(self.valid_X)
        self.valid_y_indexing_method = get_indexing_method(self.valid_y)
        self.test_X_indexing_method = get_indexing_method(self.test_X)
        self.test_y_indexing_method = get_indexing_method(self.test_y)

        self.len_labled=self.labled_dataset.__len__()
        self.len_unlabled = self.unlabled_dataset.__len__()
        self.len_test = self.test_dataset.__len__()
        self.len_valid = self.valid_dataset.__len__()
        self.data_initialized=True
        return self

    def add_transform(self,transform,dim,x,y=0):
        self.train_dataset.add_transform(transform,dim,x,y)

    def add_target_transform(self,target_transform,dim,x,y=0):
        self.train_dataset.add_target_transform(target_transform,dim,x,y)

    def add_transforms(self,transforms,dim,x,y=0):
        self.train_dataset.add_transforms(transforms, dim, x, y)

    def add_unlabled_transform(self,unlabled_transform,dim,x,y=0):
        self.train_dataset_dataset.add_unlabled_transform(unlabled_transform,dim,x,y)

    def add_valid_transform(self,valid_transform,dim,x,y=0):
        self.valid_dataset_dataset.add_transform(valid_transform,dim,x,y)

    def add_test_transform(self,test_transform,dim,x,y=0):
        self.test_dataset_dataset.add_transform(test_transform,dim,x,y)

    def get_dataset(self,train=True,test=False,valid=False,labled=True):
        if train:
            return self.train_dataset
        elif test:
            return self.test_dataset
        elif valid:
            return self.valid_dataset
        elif labled:
            return self.labled_dataset
        else:
            return self.unlabled_dataset

    def __getitem__(self, i, test=False,valid=False,labled=True):
        if test:
            i,Xi,yi=self.test_dataset[i]
        elif valid:
            i,Xi, yi = self.valid_dataset[i]
        elif labled:
            i,Xi,yi=self.labled_dataset[i]
        else:
            i,Xi,yi= self.unlabled_dataset[i]
        return i,Xi,yi

    def __len__(self,test=False,valid=False,labled=True):
        if test:
            return self.len_test
        elif valid:
            return self.len_valid
        elif labled:
            return self.len_labled
        else:
            return self.len_unlabled




