from Semi_sklearn.Data_Augmentation.AutoContrast import AutoContrast
from Semi_sklearn.Data_Augmentation.Brightness import Brightness
from Semi_sklearn.Data_Augmentation.Color import Color
from Semi_sklearn.Data_Augmentation.Contrast import Contrast
from Semi_sklearn.Data_Augmentation.Equalize import Equalize
from Semi_sklearn.Data_Augmentation.Identity import Identity
from Semi_sklearn.Data_Augmentation.Posterize import Posterize
from Semi_sklearn.Data_Augmentation.Rotate import Rotate
from Semi_sklearn.Data_Augmentation.Sharpness import Sharpness
from Semi_sklearn.Data_Augmentation.ShearX import ShearX
from Semi_sklearn.Data_Augmentation.ShearY import ShearY
from Semi_sklearn.Data_Augmentation.Solarize import Solarize
from Semi_sklearn.Data_Augmentation.TranslateX import TranslateX
from Semi_sklearn.Data_Augmentation.TranslateY import TranslateY
from Semi_sklearn.Data_Augmentation.Cutout import Cutout
from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
import random
import numpy as np


augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]



class RandAugment(Augmentation):
    def __init__(self, n, m, num_bins,augment_list=None):
        super().__init__()
        self.n = n
        self.m = m
        self.num_bins=num_bins
        self.augment_list = [(AutoContrast, None, None),
            (Brightness, 0.0, 0.9),
            (Color, 0.0, 0.9),
            (Contrast, 0.0, 0.9),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 8),
            (Rotate,  0.0,30.0),
            (Sharpness, 0.0, 0.9),
            (ShearX, 0.0, 0.3),
            (ShearY, 0.0, 0.3),
            (Solarize,  0.0,255.0),
            (TranslateX, 0.0, 150.0 / 331.0),
            (TranslateY, 0.0,150.0 / 331.0)] if augment_list is None else augment_list

    def transform(self, X):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_v, max_v in ops:
            if min_v is None and max_v is None:
                aug=op()
            else:
                aug=op(min_v=min_v,max_v=max_v,num_bins=self.num_bins,magnitude=self.m)
            X=aug.fit_transform(X)
        return X