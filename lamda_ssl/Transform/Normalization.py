import torch
from torchvision.transforms import transforms
from lamda_ssl.Transform.Transformer import Transformer
from lamda_ssl.utils import to_image,to_numpy
import numpy as np
class Normalization(Transformer):
    def __init__(self,mean=None,std=None):
        super().__init__()
        self.mean=mean
        self.std=std
        self.normalize=transforms.Normalize(mean=self.mean, std=self.std)


    def transform(self,X):
        if X is not None:
            # print(X.shape)
            # print(1)
            # print(X)
            # print(2)
            # print(self.std)
            # print(self.mean)
            X=self.normalize(X.float())
            # print(X)
            #print(X.shape)
            return X
        else:
            raise ValueError('No data to augment')