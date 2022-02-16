from Semi_sklearn.Transform.Transformer import Transformer
from torchvision import transforms
from Semi_sklearn.utils import partial
import PIL
import numpy as np
import torch

class RandomCrop(Transformer):
    def __init__(self, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        self.padding=padding
        self.pad_if_needed=pad_if_needed
        self.fill=fill
        self.padding_mode=padding_mode
        self.augmentation=partial(transforms.RandomCrop,pad_if_needed=pad_if_needed,
                                    fill=fill,padding_mode=padding_mode)
    def transform(self,X):
        if isinstance(X,np.ndarray):
            X=PIL.Image.fromarray(X)
        if isinstance(X,PIL.Image.Image):
            size=X.size[0]
            padding=int(self.padding*size) if self.padding is not None else None
            augmentation = self.augmentation(size=size, padding=padding)
            X = augmentation(X)
            return X
        elif isinstance(X,torch.Tensor):
            size=X.shape[-2]
            padding=int(self.padding*size) if self.padding is not None else None
            if len(X.shape)==4:
                for _ in range(X.shape[0]):
                    augmentation = self.augmentation(size=size, padding=padding)
                    X[_]=augmentation(X[_])
            else:
                augmentation = self.augmentation(size=size, padding=padding)
                X = augmentation(X)
            return X
        else:
            raise ValueError('No data to augment')
