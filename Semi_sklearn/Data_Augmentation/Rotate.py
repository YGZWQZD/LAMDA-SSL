import torch
import numpy as np

from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
import torchvision.transforms.functional as F
class Rotate(Augmentation):
    def __init__(self, v):
        super().__init__()
        self.v=v

    def transform(self,X):

        if X is not None:
            X=F.rotate(X,self.v)
            return X
        else:
            raise ValueError('No data to augment')