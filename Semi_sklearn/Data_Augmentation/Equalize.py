from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
import torchvision.transforms.functional as F
import PIL
import torch

class Equalize(Augmentation):
    def __init__(self):
        super().__init__()

    def transform(self,X):
        if isinstance(X, PIL.Image.Image):
            X = PIL.ImageOps.equalize(X)
            return X
        elif isinstance(X, torch.Tensor):
            X = F.equalize(X)
            return X
            #X=F.equalize(X.contiguous())
        else:
            raise ValueError('No data to augment')