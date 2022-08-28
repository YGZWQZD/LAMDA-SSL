from LAMDA_SSL.Dataset.SemiDataset import SemiDataset
from LAMDA_SSL.Base.TabularMixin import TabularMixin
from LAMDA_SSL.Split.DataSplit import DataSplit
from LAMDA_SSL.Dataset.TrainDataset import TrainDataset
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from sklearn import datasets
import numpy as np

class BreastCancer(SemiDataset,TabularMixin):
    def __init__(
        self,
        default_transforms=False,
        pre_transform=None,
        transforms=None,
        transform = None,
        target_transform = None,
        unlabeled_transform=None,
        valid_transform=None,
        test_transform=None,
        test_size=None,
        valid_size=None,
        labeled_size=0.1,
        stratified=False,
        shuffle=True,
        random_state=0,
    ) -> None:
        self.default_transforms=default_transforms
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


        self.dataset = datasets.load_breast_cancer()
        SemiDataset.__init__(self,pre_transform=pre_transform,transforms=transforms,transform=transform, target_transform=target_transform,
                             unlabeled_transform=unlabeled_transform,test_transform=test_transform,
                             valid_transform=valid_transform,labeled_size=labeled_size,test_size=test_size,
                             valid_size=valid_size,stratified=stratified,shuffle=shuffle,random_state=random_state)
        TabularMixin.__init__(self)
        if self.default_transforms:
            self.init_default_transforms()
        self.init_dataset()

    def _init_dataset(self):
        X, y = self.dataset.data, self.dataset.target.astype(np.float32)
        if self.test_size is not None:
            test_X, test_y, train_X, train_y = DataSplit(X=X, y=y,
                                               size_split=self.test_size,
                                               stratified=self.stratified,
                                               shuffle=self.shuffle,
                                               random_state=self.random_state
                                               )
        else:
            test_X = None
            test_y = None
            train_X=X
            train_y=y

        if self.valid_size is not None:
            valid_X, valid_y, train_X, train_y = DataSplit(X=train_X, y=train_y,
                                                                   size_split=self.valid_size,
                                                                   stratified=self.stratified,
                                                                   shuffle=self.shuffle,
                                                                   random_state=self.random_state
                                                                   )
        else:
            valid_X=None
            valid_y=None

        if self.labeled_size is not None:
            labeled_X, labeled_y, unlabeled_X, unlabeled_y = DataSplit(X=train_X,y=train_y,
                                                                   size_split=self.labeled_size,
                                                                   stratified=self.stratified,
                                                                   shuffle=self.shuffle,
                                                                   random_state=self.random_state
                                                                   )
        else:
            labeled_X, labeled_y=train_X,train_y
            unlabeled_X, unlabeled_y=None,None
        self.test_dataset=LabeledDataset(pre_transform=self.pre_transform,transform=self.test_transform)
        self.test_dataset.init_dataset(test_X,test_y)
        self.valid_dataset=LabeledDataset(pre_transform=self.pre_transform,transform=self.valid_transform)
        self.valid_dataset.init_dataset(valid_X,valid_y)
        self.train_dataset = TrainDataset(pre_transform=self.pre_transform,transforms=self.transforms,transform=self.transform,
                                          target_transform=self.target_transform,unlabeled_transform=self.unlabeled_transform)
        labeled_dataset=LabeledDataset(pre_transform=self.pre_transform,transforms=self.transforms,transform=self.transform,
                                          target_transform=self.target_transform)
        labeled_dataset.init_dataset(labeled_X, labeled_y)
        unlabeled_dataset=UnlabeledDataset(pre_transform=self.pre_transform,transform=self.unlabeled_transform)
        unlabeled_dataset.init_dataset(unlabeled_X, unlabeled_y)
        self.train_dataset.init_dataset(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset)