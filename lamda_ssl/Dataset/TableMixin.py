from lamda_ssl.Transform.Normalization import Normalization
from lamda_ssl.Transform.MinMaxScalar import MinMaxScalar
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from lamda_ssl.Transform.ToTensor import ToTensor
import matplotlib.pyplot as plt
class TableMixin:
    def __init__(self):
        pass

    def init_transforms(self):
        self.transforms=None
        self.target_transform=None
        # self.pre_transform=Pipeline([('StandardScaler',preprocessing.StandardScaler())
        #                       # ('Normalization',Normalization(mean=self.mean,std=self.std))
        #                       ])
        self.pre_transform=Pipeline([('StandardScaler',preprocessing.StandardScaler())
                              # ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        self.transform=Pipeline([('ToTensor', ToTensor())])
        self.unlabeled_transform=Pipeline([('ToTensor', ToTensor())])
        self.test_transform=Pipeline([('ToTensor', ToTensor())])
        self.valid_transform=Pipeline([('ToTensor', ToTensor())])
        return self