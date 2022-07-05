import torch

from LAMDA_SSL.Transform.Transformer import Transformer
import PIL
import numpy as np


class ToImage(Transformer):

    def __init__(self):
        super(ToImage, self).__init__()

    def transform(self,X):
        if isinstance(X,torch.Tensor):
            X=X.numpy()
        if len(X.shape)==3 and X.shape[0]<=3:
            X=X.transpose((1,2,0))
        if isinstance(X, np.ndarray):
            X = PIL.Image.fromarray(X)
        return X