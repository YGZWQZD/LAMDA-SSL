import numbers

import torch

from LAMDA_SSL.Base.Transformer import Transformer
import numpy as np

class Noise(Transformer):
    def __init__(self,noise_level=0.1):
        super().__init__()
        # >> Parameter:
        # >> - noise_level: the level of noise.
        self.noise_level=noise_level

    def transform(self,X):

        if isinstance(X,torch.Tensor):
            noise = np.random.normal(loc=0.0, scale=self.noise_level, size=X.size())
            noise = torch.autograd.Variable(torch.FloatTensor(noise).to(X.device))
            return X+noise
        elif isinstance(X,np.ndarray):
            noise = np.random.normal(loc=0.0, scale=self.noise_level, size=X.shape)
            return X+noise
        elif isinstance(X,numbers.Number):
            noise = np.random.normal(loc=0.0, scale=self.noise_level, size=1)
            return X+noise
        else:
            raise ValueError('No data to augment')