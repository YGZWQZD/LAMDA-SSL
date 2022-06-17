import PIL.Image
import torch
import numpy as np
from lamda_ssl.Transform.Transformer import Transformer
import torchvision.transforms.functional as F

class Solarize(Transformer):
    def __init__(self, min_v,max_v,num_bins,magnitude,v=None):
        super().__init__()
        self.max_v=max_v
        self.min_v=min_v
        self.num_bins=num_bins
        self.magnitude=magnitude
        self.magnitudes=torch.linspace(self.max_v, self.min_v, self.num_bins)
        self.v=float(self.magnitudes[self.magnitude-1].item()) if v is None else v

    def transform(self,X):
        if isinstance(X,np.ndarray):
            X=PIL.Image.fromarray(X)
        if isinstance(X,PIL.Image.Image):
            X=PIL.ImageOps.solarize(X, self.v)
            return X
        elif isinstance(X,torch.Tensor):
            X=F.solarize(X, self.v)
            return X
        else:
            raise ValueError('No data to augment')