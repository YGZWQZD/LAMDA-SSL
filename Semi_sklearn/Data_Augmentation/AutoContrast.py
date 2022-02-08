from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
import torchvision.transforms.functional as F
import PIL
import torch

class AutoContrast(Augmentation):
    def __init__(self):
        super().__init__()


    def transform(self,X):
        if isinstance(X,PIL.Image.Image):
            X=PIL.ImageOps.autocontrast(X)
            return X
        elif isinstance(X,torch.Tensor):
            X = F.autocontrast(X)
            return X
        else:
            raise ValueError('No data to augment')