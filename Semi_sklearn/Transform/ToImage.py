import torch

from Semi_sklearn.Transform.Transformer import Transformer
import PIL
import numpy as np


class ToImage(Transformer):

    def __init__(self):
        super(ToImage, self).__init__()

    def transform(self,X):
        if isinstance(X,torch.Tensor):
            X=X.numpy()
        if isinstance(X, np.ndarray):
            X = PIL.Image.fromarray(X)
        return X