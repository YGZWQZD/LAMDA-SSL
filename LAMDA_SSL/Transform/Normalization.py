import torch
from torchvision.transforms import transforms
from LAMDA_SSL.Transform.Transformer import Transformer
from LAMDA_SSL.utils import to_image,to_numpy
import numpy as np
class Normalization(Transformer):
    def __init__(self,mean=None,std=None):
        super().__init__()
        self.mean=mean
        self.std=std
        self.normalize=transforms.Normalize(mean=self.mean, std=self.std)


    def transform(self,X):
        if X is not None:
            X=self.normalize(X.float())
            return X
        else:
            raise ValueError('No data to augment')