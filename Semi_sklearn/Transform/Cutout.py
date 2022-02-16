import PIL,PIL.ImageEnhance
import torch

from Semi_sklearn.Transform.Transformer import Transformer
from Semi_sklearn.Transform.CutoutAbs import CutoutAbs
import numpy as np

class Cutout(Transformer):
    def __init__(self, v,fill):
        super().__init__()
        self.v=v
        self.fill=fill
        assert 0.0 <= v <= 0.5

    def transform(self,X):
        if isinstance(X,np.ndarray):
            X=PIL.Image.fromarray(X)
        if isinstance(X,PIL.Image.Image):
            v = self.v * X.size[0]
        elif isinstance(X,torch.Tensor):
            v = self.v * X.shape[-2]
        else:
            raise ValueError('No data to augment')
        X=CutoutAbs(v,self.fill).fit_transform(X)

        return X

