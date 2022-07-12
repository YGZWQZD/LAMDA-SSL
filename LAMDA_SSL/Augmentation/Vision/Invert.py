from LAMDA_SSL.Base.Transformer import Transformer
import torchvision.transforms.functional as F
import PIL
import torch
import numpy as np

class Invert(Transformer):
    def __init__(self):
        super().__init__()

    def transform(self,X):
        if isinstance(X,np.ndarray):
            X=PIL.Image.fromarray(X)
        if isinstance(X,PIL.Image.Image):
            X=PIL.ImageOps.invert(X)
            return X
        elif isinstance(X,torch.Tensor):
            X = F.invert(X)
            return X
        else:
            raise ValueError('No data to augment')