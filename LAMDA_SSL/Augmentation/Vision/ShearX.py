from LAMDA_SSL.Base.Transformer import Transformer
import torchvision.transforms.functional as F
import math
import random
import torch
import PIL
import numpy as np

class ShearX(Transformer):
    def __init__(self, min_v=0,max_v=0.3,num_bins=10,magnitude=5,v=None):
        # >> Parameter:
        # >> - min_v: The minimum value of the augmentation.
        # >> - max_v: The maximum value of the augmentation.
        # >> - num_bins: The number of intervals  division for the value of the augmentation.
        # >> - magnitude: The level of the augmentation.
        # >> - v: Specify the value of the augmentation directly.
        super().__init__()
        self.max_v=max_v
        self.min_v=min_v
        self.num_bins=num_bins
        self.magnitude=magnitude
        self.magnitudes=torch.linspace(min_v, max_v, num_bins)
        self.v=float(self.magnitudes[self.magnitude].item())if v is None else v

    def transform(self,X):
        if isinstance(X,np.ndarray):
            X=PIL.Image.fromarray(X)
        if isinstance(X,PIL.Image.Image):
            _v = self.v if random.random() < 0.5 else self.v * -1
            X=X.transform(X.size, PIL.Image.AFFINE, (1, _v, 0, 0, 1, 0))
            return X
        elif isinstance(X,torch.Tensor):
            if len(X.shape)==4:
                for _ in range(X.shape[0]):
                    _v = self.v if random.random() < 0.5 else self.v * -1
                    X[_]=F.affine(X[_],angle=0,translate=[0, 0],scale=1.0,shear=[math.degrees(_v),0.0])
            else:
                _v = self.v if random.random() < 0.5 else self.v * -1
                X=F.affine(X,angle=0,translate=[0, 0],scale=1.0,shear=[math.degrees(_v),0.0])
            return X
        else:
            raise ValueError('No data to augment')