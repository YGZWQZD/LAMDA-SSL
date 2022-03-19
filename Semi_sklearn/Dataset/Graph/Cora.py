import numpy as np
from Semi_sklearn.Dataset.SemiDataset import SemiDataset
from Semi_sklearn.Split.SemiSplit import SemiSplit
from Semi_sklearn.Dataset.TrainDataset import TrainDataset
from Semi_sklearn.Dataset.LabeledDataset import LabeledDataset
from Semi_sklearn.Dataset.UnlabeledDataset import UnlabeledDataset
import torch
from Semi_sklearn.Dataset.GraphMixin import GraphMixin
from torch_geometric.datasets.planetoid import Planetoid
class Cora(SemiDataset,GraphMixin):
    # name='Cora'
    # url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    # names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    # processed_file_names='data.pt'
    def __init__(
        self,
        root: str,
        transforms=None,
        transform = None,
        target_transform = None,
        unlabeled_transform=None,
        valid_transform=None,
        test_transform=None,
        test_size=None,
        valid_size=None,
        labeled_size=None,
        stratified=False,
        shuffle=True,
        random_state=None
    ) -> None:

        self.labeled_X=None
        self.labeled_y=None
        self.unlabeled_X=None
        self.unlabeled_y=None
        self.valid_X=None
        self.valid_y=None
        self.test_X=None
        self.test_y=None

        self.labeled_dataset=None
        self.unlabeled_dataset=None
        self.train_dataset=None
        self.valid_dataset = None
        self.test_dataset=None

        self.data_initialized=False

        self.len_test=None
        self.len_valid = None
        self.len_labeled=None
        self.len_unlabeled=None

        self.labeled_X_indexing_method=None
        self.labeled_y_indexing_method =None
        self.unlabeled_X_indexing_method =None
        self.unlabeled_y_indexing_method =None
        self.valid_X_indexing_method=None
        self.valid_indexing_method=None
        self.test_X_indexing_method=None
        self.test_y_indexing_method=None
        self.root=root
        self.dataset = Planetoid(root=root, name='Cora')

        SemiDataset.__init__(self,transforms=transforms,transform=transform, target_transform=target_transform,
                             unlabeled_transform=unlabeled_transform,test_transform=test_transform,
                             valid_transform=valid_transform,labeled_size=labeled_size,test_size=test_size,valid_size=valid_size,
                             stratified=stratified,shuffle=shuffle,random_state=random_state)





    def _init_dataset(self):
        self.data=self.dataset.data

        self.data.train_mask.fill_(True)
        train_X=np.arange(len(self.data.y))
        train_y=self.data.y

        if self.test_size is not None:
            test_X, test_y, train_ind, train_y = SemiSplit(X=train_X, y=train_y,
                                                            labeled_size=self.test_size,
                                                            stratified=self.stratified,
                                                            shuffle=self.shuffle,
                                                            random_state=self.random_state
                                                            )
        else:
            test_X=None
            test_y=None

        if self.valid_size is not None:
            valid_X, valid_y, train_X, train_y = SemiSplit(X=train_X, y=train_y,
                                                            labeled_size=self.valid_size,
                                                            stratified=self.stratified,
                                                            shuffle=self.shuffle,
                                                            random_state=self.random_state
                                                            )
        else:
            valid_X=None
            valid_y=None

        if self.labeled_size is not None:
            labeled_X, labeled_y, unlabeled_X, unlabeled_y = SemiSplit(X=train_X, y=train_y,
                                                            labeled_size=self.labeled_size,
                                                            stratified=self.stratified,
                                                            shuffle=self.shuffle,
                                                            random_state=self.random_state
                                                            )
        else:
            labeled_X=train_X
            labeled_y=train_y
            unlabeled_X=None
            unlabeled_y=None

        self.data.unlabeled_mask = torch.zeros((len(self.data.y),), dtype=torch.bool)
        self.data.labeled_mask = torch.zeros((len(self.data.y),), dtype=torch.bool)
        self.data.train_mask.fill_(True)
        self.data.unlabeled_mask.fill_(False)
        self.data.labeled_mask.fill_(False)
        self.data.test_mask.fill_(False)
        self.data.val_mask.fill_(False)
        if unlabeled_X is not None:
            self.data.unlabeled_mask[unlabeled_X] = True
        if labeled_X is not None:
            self.data.labeled_mask[labeled_X] = True
        if valid_X is not None:
            self.data.val_mask[valid_X]=True
        if test_X is not None:
            self.data.test_mask[test_X] = True


        self.data.train_mask[self.data.val_mask|self.data.test_mask] = False


        self.test_dataset=LabeledDataset(transform=self.test_transform)
        self.test_dataset.init_dataset(test_X,test_y)
        self.valid_dataset=LabeledDataset(transform=self.valid_transform)
        self.valid_dataset.init_dataset(valid_X,valid_y)
        self.train_dataset = TrainDataset(transforms=self.transforms,transform=self.transform,
                                          target_transform=self.target_transform,unlabeled_transform=self.unlabeled_transform)
        labeled_dataset=LabeledDataset(transforms=self.transforms,transform=self.transform,
                                          target_transform=self.target_transform)
        labeled_dataset.init_dataset(labeled_X, labeled_y)
        unlabeled_dataset=UnlabeledDataset(transform=self.unlabeled_transform)
        unlabeled_dataset.init_dataset(unlabeled_X, unlabeled_y)
        self.train_dataset.init_dataset(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset)




