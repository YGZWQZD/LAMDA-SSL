import torch

from LAMDA_SSL.Base.Transformer import Transformer
import PIL
import numpy as np


class ToImage(Transformer):

    def __init__(self,channels=3,channels_first=False):
        # > - Parameter:
        # >> - channels: The number of channels of input images.
        # >> - channels_first: Whether the number of channels is before the image size.
        super(ToImage, self).__init__()
        self.channels=channels
        self.channels_first=channels_first

    def transform(self,X):
        if isinstance(X,torch.Tensor):
            X=X.numpy()
        if self.channels_first is True and len(X.shape)==3:
            X=X.transpose((1,2,0))
        if isinstance(X, np.ndarray):
            X = PIL.Image.fromarray(X)
        return X