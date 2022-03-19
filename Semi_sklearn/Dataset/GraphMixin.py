from Semi_sklearn.Transform.Normalization import Normalization
from Semi_sklearn.Transform.NormalizeFeatures import NormalizeFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
class GraphMixin:
    def __init__(self):
        pass

    def init_transforms(self):
        self.transforms=None
        self.target_transform=None
        self.transform=Pipeline([('NormalizeFeatures',NormalizeFeatures()),
                              # ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        self.unlabeled_transform=Pipeline([('NormalizeFeatures',NormalizeFeatures()),
                              # ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        self.test_transform=Pipeline([('NormalizeFeatures',NormalizeFeatures()),
                              # ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        self.valid_transform=Pipeline([('NormalizeFeatures',NormalizeFeatures()),
                              # ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        return self