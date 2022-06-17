from lamda_ssl.Transform.NormalizeFeatures import NormalizeFeatures
from sklearn.pipeline import Pipeline

class GraphMixin:
    def __init__(self):
        pass

    def init_transforms(self):
        self.transforms=None
        self.target_transform=None
        self.transform=Pipeline([('NormalizeFeatures',NormalizeFeatures())
                              ])
        self.unlabeled_transform=Pipeline([('NormalizeFeatures',NormalizeFeatures())
                              ])
        self.test_transform=Pipeline([('NormalizeFeatures',NormalizeFeatures())
                              ])
        self.valid_transform=Pipeline([('NormalizeFeatures',NormalizeFeatures())
                              ])
        return self