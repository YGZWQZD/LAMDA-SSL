from Semi_sklearn.Transform.Transformer import Transformer
import torchvision.transforms.functional as F
import torch
import PIL
import numpy as np

class Posterize(Transformer):
    def __init__(self, min_v,max_v,num_bins,magnitude,v=None):
        super().__init__()
        self.max_v=max_v
        self.min_v=min_v
        self.num_bins=num_bins
        self.magnitude=magnitude
        self.magnitudes=self.max_v - (torch.arange(self.num_bins) / ((self.num_bins - 1) / self.min_v)).round().int()
        self.v=self.magnitudes[self.magnitude-1].item()if v is None else v
        self.v = int(max(1, self.v))


    def transform(self,X):
        if isinstance(X,np.ndarray):
            X=PIL.Image.fromarray(X)
        if isinstance(X,PIL.Image.Image):
            X=PIL.ImageOps.posterize(X, self.v)
            return X
        elif isinstance(X,torch.Tensor):
            X=F.posterize(X, int(self.v))
            return X
        else:
            raise ValueError('No data to augment')
