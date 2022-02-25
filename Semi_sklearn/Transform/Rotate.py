import torch
import numpy as np
import PIL
from Semi_sklearn.Transform.Transformer import Transformer
import torchvision.transforms.functional as F
import random

class Rotate(Transformer):
    def __init__(self, min_v=None,max_v=None,num_bins=None,magnitude=None,v=None):
        super().__init__()
        self.max_v=max_v
        self.min_v=min_v
        self.num_bins=num_bins
        self.magnitude=magnitude
        if v is None:
            self.magnitudes=torch.linspace(min_v, max_v, num_bins)
            self.v = float(self.magnitudes[self.magnitude].item())
        else:
            self.v=v

    def transform(self,X,rand=False):
        if isinstance(X,np.ndarray):
            X=PIL.Image.fromarray(X)
        if isinstance(X,PIL.Image.Image):
            _v=self.v if random.random() < 0.5 else self.v*-1
            X=X.rotate(_v)
            return X
        elif isinstance(X,torch.Tensor):
            if len(X.shape)==4:
                for _ in range(X.shape[0]):
                    _v = self.v if random.random() < 0.5 else self.v * -1
                    X[_]=F.rotate(X[_],_v)
            else:
                _v = self.v if random.random() < 0.5 else self.v * -1
                X = F.rotate(X,_v)
            return X

        else:
            raise ValueError('No data to augment')