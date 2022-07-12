from LAMDA_SSL.Base.Transformer import Transformer
import torchvision.transforms.functional as F
import PIL
import torch
import numpy as np

class Equalize(Transformer):
    def __init__(self,scale=255):
        # >> Parameter:
        # >> - scale: Scale of image pixel values
        super().__init__()
        self.scale=scale

    def transform(self,X):
        if isinstance(X,np.ndarray):
            X=PIL.Image.fromarray(X)
        if isinstance(X, PIL.Image.Image):
            X = PIL.ImageOps.equalize(X)
            return X
        elif isinstance(X, torch.Tensor):
            X = F.equalize((X * self.scale).type(torch.uint8)) / self.scale
            return X
        else:
            raise ValueError('No data to augment')