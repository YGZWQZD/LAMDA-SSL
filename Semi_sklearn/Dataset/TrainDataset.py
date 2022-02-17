from torch.utils.data import Dataset
from .LabledDataset import LabledDataset
from Semi_sklearn.Dataset.UnlabledDataset import UnlabledDataset
from ..Split.SemiSplit import SemiSplit

class TrainDataset(Dataset):
    def __init__(self,
                 transforms=None,
                 transform=None,
                 target_transform=None,
                 unlabled_transform=None,
                 labled_size=None,
                 stratified=False,
                 shuffle=True,
                 random_state=None):

        self.transforms=transforms
        self.transform = transform
        self.target_transform=target_transform
        self.unlabled_transform = unlabled_transform

        self.labled_size=labled_size
        self.stratified=stratified
        self.shuffle=shuffle
        self.random_state=random_state

        self.labled_X=None
        self.labled_y=None
        self.unlabled_X=None
        self.unlabled_y=None
        self.len_labled=None
        self.len_unlabled=None
        self.labled_dataset=LabledDataset(transforms=self.transforms,transform=self.unlabled_transform,
                                          target_transform=self.target_transform)
        self.unlabled_dataset=UnlabledDataset(transform=self.unlabled_transform)
        self.data_initialized=False


    def _init_dataset(self):
        raise NotImplementedError(
            "_init_dataset method of SemiTrainDataset class must be implemented."
        )

    def init_dataset(self,labled_X=None,labled_y=None,unlabled_X=None,
                    unlabled_y=None,labled_dataset=None,unlabled_dataset=None):
        if labled_X is not None:
            if unlabled_X is None and self.labled_size is not None:
                labled_X,labled_y,unlabled_X,unlabled_y=SemiSplit(X=labled_X,y=labled_y,
                                                                labled_size=self.labled_size,
                                                                stratified=self.stratified,
                                                                shuffle=self.shuffle,
                                                                random_state=self.random_state
                                                                )
            self.unlabled_dataset.init_dataset(unlabled_X, unlabled_y)
            self.labled_dataset.init_dataset(labled_X,labled_y)

        elif labled_dataset is not None:
            if unlabled_dataset is not None:
                self.unlabled_dataset=unlabled_dataset
                self.labled_dataset=labled_dataset
            elif self.labled_size is not None:
                labled_X=getattr(labled_dataset,'X')
                labled_y=getattr(labled_dataset,'y')
                labled_X,labled_y,unlabled_X,unlabled_y=SemiSplit(X=labled_X,y=labled_y,
                                                                labled_size=self.labled_size,
                                                                stratified=self.stratified,
                                                                shuffle=self.shuffle,
                                                                random_state=self.random_state,
                                                                )
                self.unlabled_dataset.init_dataset(unlabled_X, unlabled_y)
                self.labled_dataset.init_dataset(labled_X, labled_y)
        else:
            self._init_dataset()

        self.labled_X = getattr(self.labled_dataset,'X')
        self.labled_y = getattr(self.labled_dataset,'y')
        self.unlabled_X = getattr(self.unlabled_dataset,'X')
        self.unlabled_y = getattr(self.unlabled_dataset,'y')
        self.len_labled=self.labled_dataset.__len__()
        self.len_unlabled = self.unlabled_dataset.__len__()
        self.data_initialized=True
        return self

    def add_transform(self,transform,dim,x,y):
        self.labled_dataset.add_transform(transform,dim,x,y=0)

    def add_target_transform(self,target_transform,dim,x,y=0):
        self.labled_dataset.add_target_transform(target_transform,dim,x,y)

    def add_transforms(self,transforms,dim,x,y=0):
        self.labled_dataset.add_transforms(transforms, dim, x, y)

    def add_unlabled_transform(self,unlabled_transform,dim,x,y=0):
        self.unlabled_dataset.add_transform(unlabled_transform,dim,x,y)

    def get_dataset(self,labled):
        if labled:
            return self.labled_dataset
        else:
            return self.unlabled_dataset

    def __getitem__(self, i, labled=True):
        if labled:
            i,Xi,yi=self.labled_dataset[i]
        else:
            i,Xi,yi=self.unlabled_dataset[i]
        return i, Xi, yi

    def __len__(self,labled=True):
        if labled:
            return self.len_labled
        else:
            return self.len_unlabled





