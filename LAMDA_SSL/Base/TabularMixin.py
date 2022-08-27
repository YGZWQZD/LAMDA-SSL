from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from LAMDA_SSL.Transform.ToTensor import ToTensor

class TabularMixin:
    def __init__(self):
        pass

    def init_default_transforms(self):
        # >> init_default_transform: Initialize the default data transformation method.
        self.transforms=None
        self.target_transform=None
        self.pre_transform=Pipeline([('StandardScaler',preprocessing.StandardScaler())
                              ])
        self.transform=Pipeline([('ToTensor', ToTensor())])
        self.unlabeled_transform=Pipeline([('ToTensor', ToTensor())])
        self.test_transform=Pipeline([('ToTensor', ToTensor())])
        self.valid_transform=Pipeline([('ToTensor', ToTensor())])
        return self