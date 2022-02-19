from torch.utils.data import Dataset
from .TrainDataset import TrainDataset
from .LabeledDataset import LabeledDataset
from .UnlabeledDataset import UnlabeledDataset
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
                 unlabeled_transform=None,
                 valid_transform=None,
                 test_transform=None,
                 test_size=None,
                 valid_size=None,
                 labeled_size=None,
                 stratified=False,
                 shuffle=True,
                 random_state=None):
        self.transforms=transforms
        self.transform = transform
        self.target_transform=target_transform
        self.unlabeled_transform = unlabeled_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform

        self.labeled_size=labeled_size
        self.valid_size=valid_size
        self.test_size = test_size

        self.stratified=stratified
        self.shuffle=shuffle
        self.random_state=random_state

        self.labeled_X=None
        self.labeled_y=None
        self.unlabeled_X=None
        self.unlabeled_y=None
        self.valid_X=None
        self.valid_y=None
        self.test_X=None
        self.test_y=None

        self.labeled_dataset=LabeledDataset(transforms=self.transforms, transform=self.transform,
                                          target_transform=self.target_transform)
        self.unlabeled_dataset=UnlabeledDataset(transform=self.unlabeled_transform)
        self.train_dataset = TrainDataset(transforms=self.transforms, transform=self.transform,
                                          target_transform=self.target_transform,
                                          unlabeled_transform=self.unlabeled_transform,
                                          labeled_size=self.labeled_size, stratified=self.stratified,
                                          shuffle=self.shuffle, random_state=self.random_state)
        self.valid_dataset = LabeledDataset(transform=self.valid_transform)
        self.test_dataset=LabeledDataset(transform=self.test_transform)

        self.len_test=None
        self.len_valid = None
        self.len_labeled=None
        self.len_unlabeled=None

        self.labeled_X_indexing_method=None
        self.labeled_y_indexing_method =None
        self.unlabeled_X_indexing_method =None
        self.unlabeled_y_indexing_method =None
        self.test_X_indexing_method=None
        self.test_y_indexing_method=None
        self.valid_X_indexing_method=None
        self.valid_y_indexing_method=None

        self.data_initialized=False
        # self.labeled_class=LabeledDataset
        # self.unLabeled_class=UnlabeledDataset
        # self.train_class=partial(TrainDataset,labeled_size=self.labeled_size,stratified=self.stratified,
        #                              shuffle=self.shuffle,random_state=self.random_state)

    def _init_dataset(self):
        raise NotImplementedError(
            "_init_dataset method of SemiDataset class must be implemented."
        )

    def init_dataset(self,labeled_X=None,labeled_y=None,unlabeled_X=None,
                    unlabeled_y=None,test_X=None,test_y=None,valid_X=None,valid_y=None,train_dataset=None,
                    test_dataset=None,valid_dataset=None,labeled_dataset=None,unlabeled_dataset=None):

        if labeled_X is None and labeled_dataset is None and train_dataset is None:
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
            if labeled_dataset is not None:
                labeled_X = getattr(labeled_dataset, 'X')
                labeled_y = getattr(labeled_dataset, 'y')

            if test_dataset is not None:
                self.test_dataset=test_dataset
                # setattr(self.test_dataset,'transform',self.test_transform)
            elif test_X is not None:
                self.test_dataset.inin_dataset(test_X,test_y)
            elif self.test_size is not None:
                test_X, test_y,labeled_X, labeled_y = SemiSplit(X=labeled_X, y=labeled_y,
                                                           labeled_size=self.test_size,
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
                if labeled_dataset is not None:
                    labeled_X=getattr(labeled_dataset,'X')
                    labeled_y=getattr(labeled_dataset,'y')

                valid_X, valid_y,labeled_X, labeled_y = SemiSplit(X=labeled_X, y=labeled_y,
                                                           labeled_size=self.valid_size,
                                                           stratified=self.stratified,
                                                           shuffle=self.shuffle,
                                                           random_state=self.random_state
                                                        )
                self.valid_dataset.inin_dataset(self.test_X, self.test_y)


            self.train_dataset.init_dataset(labeled_X=labeled_X,labeled_y=labeled_y,unlabeled_X=unlabeled_X,
                    unlabeled_y=unlabeled_y,labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset)

        self.labeled_dataset=self.train_dataset.get_dataset(labeled=True)
        self.unlabeled_dataset=self.train_dataset.get_dataset(labeled=False)

        self.test_X = getattr(self.test_dataset,'X')
        self.test_y = getattr(self.test_dataset,'y')
        self.valid_X = getattr(self.valid_dataset,'X')
        self.valid_y = getattr(self.valid_dataset,'y')
        self.labeled_X=getattr(self.labeled_dataset,'X')
        self.labeled_y = getattr(self.labeled_dataset,'y')
        self.unlabeled_X = getattr(self.unlabeled_dataset,'X')
        self.unlabeled_y = getattr(self.unlabeled_dataset,'y')

        self.labeled_X_indexing_method=get_indexing_method(self.labeled_X)
        self.labeled_y_indexing_method = get_indexing_method(self.labeled_y)
        self.unlabeled_X_indexing_method =get_indexing_method(self.unlabeled_X)
        self.unlabeled_y_indexing_method = get_indexing_method(self.unlabeled_y)
        self.valid_X_indexing_method = get_indexing_method(self.valid_X)
        self.valid_y_indexing_method = get_indexing_method(self.valid_y)
        self.test_X_indexing_method = get_indexing_method(self.test_X)
        self.test_y_indexing_method = get_indexing_method(self.test_y)

        self.len_labeled=self.labeled_dataset.__len__()
        self.len_unlabeled = self.unlabeled_dataset.__len__()
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

    def add_unlabeled_transform(self,unlabeled_transform,dim,x,y=0):
        self.train_dataset_dataset.add_unlabeled_transform(unlabeled_transform,dim,x,y)

    def add_valid_transform(self,valid_transform,dim,x,y=0):
        self.valid_dataset_dataset.add_transform(valid_transform,dim,x,y)

    def add_test_transform(self,test_transform,dim,x,y=0):
        self.test_dataset_dataset.add_transform(test_transform,dim,x,y)

    def get_dataset(self,train=True,test=False,valid=False,labeled=True):
        if train:
            return self.train_dataset
        elif test:
            return self.test_dataset
        elif valid:
            return self.valid_dataset
        elif labeled:
            return self.labeled_dataset
        else:
            return self.unlabeled_dataset

    def __getitem__(self, i, test=False,valid=False,labeled=True):
        if test:
            i,Xi,yi=self.test_dataset[i]
        elif valid:
            i,Xi, yi = self.valid_dataset[i]
        elif labeled:
            i,Xi,yi=self.labeled_dataset[i]
        else:
            i,Xi,yi= self.unlabeled_dataset[i]
        return i,Xi,yi

    def __len__(self,test=False,valid=False,labeled=True):
        if test:
            return self.len_test
        elif valid:
            return self.len_valid
        elif labeled:
            return self.len_labeled
        else:
            return self.len_unlabeled




