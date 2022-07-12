from LAMDA_SSL.Base.Transformer import Transformer
import torchvision.transforms.functional as F
import torch
import PIL
import numpy as np

class Posterize(Transformer):
    def __init__(self, min_v=4,max_v=8,num_bins=10,magnitude=5,v=None,scale=255):
        # >> Parameter:
        # >> - min_v: The minimum value of the augmentation.
        # >> - max_v: The maximum value of the augmentation.
        # >> - num_bins: The number of intervals  division for the value of the augmentation.
        # >> - magnitude: The level of the augmentation.
        # >> - v: Specify the value of the augmentation directly.
        # >> - scale: Scale of image pixel values
        super().__init__()
        self.max_v=max_v
        self.min_v=min_v
        self.num_bins=num_bins
        self.magnitude=magnitude
        self.magnitudes=self.max_v - (torch.arange(self.num_bins) / ((self.num_bins - 1) / self.min_v)).round().int()
        self.v=self.magnitudes[self.magnitude].item()if v is None else v
        self.scale=scale
        self.v = int(max(1, self.v))


    def transform(self,X):
        if isinstance(X,np.ndarray):
            X=PIL.Image.fromarray(X)
        if isinstance(X,PIL.Image.Image):
            X=PIL.ImageOps.posterize(X, self.v)
            return X
        elif isinstance(X,torch.Tensor):
            X=F.posterize((X*self.scale).type(torch.uint8), self.v)/self.scale
            return X
        else:
            raise ValueError('No data to augment')
