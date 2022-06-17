from lamda_ssl.Transform.Transformer import Transformer
import torchvision.transforms.functional as F
import PIL
import torch
import numpy as np

class Equalize(Transformer):
    def __init__(self):
        super().__init__()

    def transform(self,X):
        if isinstance(X,np.ndarray):
            X=PIL.Image.fromarray(X)
        if isinstance(X, PIL.Image.Image):
            X = PIL.ImageOps.equalize(X)
            return X
        elif isinstance(X, torch.Tensor):
            X = F.equalize(X)
            return X
            #X=F.equalize(X.contiguous())
        else:
            raise ValueError('No data to augment')