from LAMDA_SSL.Base.Transformer import Transformer
from torchvision import transforms
from LAMDA_SSL.utils import partial
import PIL
import numpy as np
import torch

class CenterCrop(Transformer):
    def __init__(self):
        # >> Parameter:
        # >> - padding: Optional padding on each border of the image. Default is None. If a single int is provided this is used to pad all borders. If sequence of length 2 is provided this is the padding on left/right and top/bottom respectively. If a sequence of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
        # >> - pad_if_needed: It will pad the image if smaller than the desired size to avoid raising an exception. Since cropping is done after padding, the padding seems to be done at a random offset.
        # >> - fill: Pixel fill value for constant fill. Default is 0. If a tuple of length 3, it is used to fill R, G, B channels respectively. This value is only used when the padding_mode is constant. Only number is supported for torch Tensor. Only int or str or tuple value is supported for PIL Image.
        # >> - padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
        super().__init__()
        #self.padding=padding
        #self.pad_if_needed=pad_if_needed
        #self.fill=fill
        #self.padding_mode=padding_mode
        self.augmentation=transforms.CenterCrop
                                    #fill=fill,padding_mode=padding_mode)
    def transform(self,X):
        if isinstance(X,np.ndarray):
            X=PIL.Image.fromarray(X)
        if isinstance(X,PIL.Image.Image):
            size=X.size[0]
            #padding=int(self.padding*size) if self.padding is not None else None
            augmentation = self.augmentation(size=size)
            X = augmentation(X)
            return X
        elif isinstance(X,torch.Tensor):
            size=X.shape[-2]
            #padding=int(self.padding*size) if self.padding is not None else None
            if len(X.shape)==4:
                for _ in range(X.shape[0]):
                    augmentation = self.augmentation(size=size)
                    X[_]=augmentation(X[_])
            else:
                augmentation = self.augmentation(size=size)
                X = augmentation(X)
            return X
        else:
            raise ValueError('No data to augment')
