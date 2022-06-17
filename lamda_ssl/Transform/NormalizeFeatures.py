import torch
from torchvision.transforms import transforms
from lamda_ssl.Transform.Transformer import Transformer
from lamda_ssl.utils import to_image,to_numpy
import numpy as np
import torch_geometric.transforms as gt
class NormalizeFeatures(Transformer):
    def __init__(self,attrs=["x"]):
        super().__init__()
        self.attrs=attrs
        self.normalize=gt.NormalizeFeatures(attrs)


    def transform(self,X):
        if X is not None:
            # print(X.shape)
            # print(1)
            # print(X)
            # print(2)
            # print(self.std)
            # print(self.mean)
            X=self.normalize(X)
            # print(X)
            #print(X.shape)
            return X
        else:
            raise ValueError('No data to augment')