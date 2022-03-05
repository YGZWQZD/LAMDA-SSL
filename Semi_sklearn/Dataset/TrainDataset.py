from torch.utils.data import Dataset
from .LabeledDataset import LabeledDataset
from Semi_sklearn.Dataset.UnlabeledDataset import UnlabeledDataset
from ..Split.SemiSplit import SemiSplit

class TrainDataset(Dataset):
    def __init__(self,
                 transforms=None,
                 transform=None,
                 target_transform=None,
                 unlabeled_transform=None,
                 labeled_size=None,
                 stratified=False,
                 shuffle=True,
                 random_state=None,
                 labeled_dataset=None,
                 unlabeled_dataset=None
                 ):

        self.transforms=transforms
        self.transform = transform
        self.target_transform=target_transform
        self.unlabeled_transform = unlabeled_transform

        self.labeled_size=labeled_size
        self.stratified=stratified
        self.shuffle=shuffle
        self.random_state=random_state

        self.labeled_X=None
        self.labeled_y=None
        self.unlabeled_X=None
        self.unlabeled_y=None
        self.len_labeled=None
        self.len_unlabeled=None
        self.labeled_dataset=LabeledDataset(transforms=self.transforms,transform=self.unlabeled_transform,
                                          target_transform=self.target_transform) if labeled_dataset is None else labeled_dataset
        self.unlabeled_dataset=UnlabeledDataset(transform=self.unlabeled_transform)if unlabeled_dataset is None else unlabeled_dataset
        self.data_initialized=False


    def _init_dataset(self):
        raise NotImplementedError(
            "_init_dataset method of SemiTrainDataset class must be implemented."
        )

    def init_dataset(self,labeled_X=None,labeled_y=None,unlabeled_X=None,
                    unlabeled_y=None,labeled_dataset=None,unlabeled_dataset=None):
        if labeled_X is not None:
            if unlabeled_X is None and self.labeled_size is not None:
                labeled_X,labeled_y,unlabeled_X,unlabeled_y=SemiSplit(X=labeled_X,y=labeled_y,
                                                                labeled_size=self.labeled_size,
                                                                stratified=self.stratified,
                                                                shuffle=self.shuffle,
                                                                random_state=self.random_state
                                                                )
            self.unlabeled_dataset.init_dataset(unlabeled_X, unlabeled_y)
            self.labeled_dataset.init_dataset(labeled_X,labeled_y)

        elif labeled_dataset is not None:
            if unlabeled_dataset is not None:
                self.unlabeled_dataset=unlabeled_dataset
                self.labeled_dataset=labeled_dataset
            elif self.labeled_size is not None:
                labeled_X=getattr(labeled_dataset,'X')
                labeled_y=getattr(labeled_dataset,'y')
                labeled_X,labeled_y,unlabeled_X,unlabeled_y=SemiSplit(X=labeled_X,y=labeled_y,
                                                                labeled_size=self.labeled_size,
                                                                stratified=self.stratified,
                                                                shuffle=self.shuffle,
                                                                random_state=self.random_state,
                                                                )
                self.unlabeled_dataset.init_dataset(unlabeled_X, unlabeled_y)
                self.labeled_dataset.init_dataset(labeled_X, labeled_y)
        else:
            self._init_dataset()

        self.labeled_X = getattr(self.labeled_dataset,'X')
        self.labeled_y = getattr(self.labeled_dataset,'y')
        self.unlabeled_X = getattr(self.unlabeled_dataset,'X')
        self.unlabeled_y = getattr(self.unlabeled_dataset,'y')
        self.len_labeled=self.labeled_dataset.__len__()
        self.len_unlabeled = self.unlabeled_dataset.__len__()
        self.data_initialized=True
        return self

    def add_transform(self,transform,dim,x,y):
        self.labeled_dataset.add_transform(transform,dim,x,y=0)

    def add_target_transform(self,target_transform,dim,x,y=0):
        self.labeled_dataset.add_target_transform(target_transform,dim,x,y)

    def add_transforms(self,transforms,dim,x,y=0):
        self.labeled_dataset.add_transforms(transforms, dim, x, y)

    def add_unlabeled_transform(self,unlabeled_transform,dim,x,y=0):
        self.unlabeled_dataset.add_transform(unlabeled_transform,dim,x,y)

    def get_dataset(self,labeled):
        if labeled:
            return self.labeled_dataset
        else:
            return self.unlabeled_dataset

    def __getitem__(self, i, labeled=True):
        if labeled:
            i,Xi,yi=self.labeled_dataset[i]
        else:
            i,Xi,yi=self.unlabeled_dataset[i]
        return i, Xi, yi

    def __len__(self,labeled=True):
        if labeled:
            return self.len_labeled
        else:
            return self.len_unlabeled





