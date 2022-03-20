from Semi_sklearn.Dataset.SemiDataset import SemiDataset
from Semi_sklearn.Dataset.VisionMixin import VisionMixin

from Semi_sklearn.Transform.ImageToTensor import ToTensor
from sklearn.pipeline import Pipeline

from Semi_sklearn.Transform.ToImage import ToImage
from Semi_sklearn.Split.Split import SemiSplit
from Semi_sklearn.Dataset.TrainDataset import TrainDataset
from Semi_sklearn.Dataset.LabeledDataset import LabeledDataset
from Semi_sklearn.Dataset.UnlabeledDataset import UnlabeledDataset
from torchvision.datasets import mnist
from Semi_sklearn.Transform.MinMaxScalar import MinMaxScalar

class Mnist(SemiDataset,VisionMixin):
    def __init__(
        self,
        root: str,
        transforms=None,
        transform = None,
        target_transform = None,
        unlabeled_transform=None,
        valid_transform=None,
        test_transform=None,
        valid_size=None,
        labeled_size=0.1,
        stratified=False,
        shuffle=True,
        random_state=None,
        download: bool = False,
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
        self.min_val=0
        self.max_val=255

        self.train_data=mnist.MNIST(root=root,download=download,train=True)
        self.test_data=mnist.MNIST(root=root,download=False,train=False)
        SemiDataset.__init__(self,transforms=transforms,transform=transform, target_transform=target_transform,
                             unlabeled_transform=unlabeled_transform,test_transform=test_transform,
                             valid_transform=valid_transform,labeled_size=labeled_size,valid_size=valid_size,
                             stratified=stratified,shuffle=shuffle,random_state=random_state)
        VisionMixin.__init__(self)

    def init_transforms(self):
        self.transforms=None
        self.target_transform=None
        self.pre_transform=ToImage()
        self.transform=Pipeline([('ToTensor',ToTensor()),
                              ('MinMaxScalar',MinMaxScalar(min_val=self.min_val,max_val=self.max_val))
                              ])
        self.unlabeled_transform=Pipeline([('ToTensor',ToTensor()),
                              ('MinMaxScalar',MinMaxScalar(min_val=self.min_val,max_val=self.max_val))
                              ])
        self.test_transform=Pipeline([('ToTensor',ToTensor()),
                              ('MinMaxScalar',MinMaxScalar(min_val=self.min_val,max_val=self.max_val))
                              ])
        self.valid_transform=Pipeline([('ToTensor',ToTensor()),
                              ('MinMaxScalar',MinMaxScalar(min_val=self.min_val,max_val=self.max_val))
                              ])
        return self

    def _init_dataset(self):

        test_X, test_y = self.test_data.data, self.test_data.targets
        train_X, train_y = self.train_data.data, self.train_data.targets
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
            labeled_X, labeled_y, unlabeled_X, unlabeled_y = SemiSplit(X=train_X,y=train_y,
                                                                   labeled_size=self.labeled_size,
                                                                   stratified=self.stratified,
                                                                   shuffle=self.shuffle,
                                                                   random_state=self.random_state
                                                                   )
        else:
            labeled_X, labeled_y=train_X,train_y
            unlabeled_X, unlabeled_y=None,None
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