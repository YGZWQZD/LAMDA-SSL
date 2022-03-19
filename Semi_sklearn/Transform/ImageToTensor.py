import PIL.Image
import torch
import numpy as np

from Semi_sklearn.Transform.Transformer import Transformer
from torchvision import transforms
class ToTensor(Transformer):
    def __init__(self):
        super().__init__()

    def transform(self,X):
        if isinstance(X,np.ndarray):
            X=PIL.Image.fromarray(X)
        if isinstance(X,PIL.Image.Image):
            # print(X.size)
            X=transforms.ToTensor()(X)
            X = X.detach().float()
            # print(X)
            return X

        elif isinstance(X,torch.Tensor):
            X = X.float()
            return X
        else:
            X=torch.FloatTensor(X)
            return X
