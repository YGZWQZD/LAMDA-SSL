from LAMDA_SSL.Augmentation.Vision.AutoContrast import AutoContrast
from LAMDA_SSL.Augmentation.Vision.Brightness import Brightness
from LAMDA_SSL.Augmentation.Vision.Color import Color
from LAMDA_SSL.Augmentation.Vision.Contrast import Contrast
from LAMDA_SSL.Augmentation.Vision.Equalize import Equalize
from LAMDA_SSL.Augmentation.Vision.Identity import Identity
from LAMDA_SSL.Augmentation.Vision.Posterize import Posterize
from LAMDA_SSL.Augmentation.Vision.Rotate import Rotate
from LAMDA_SSL.Augmentation.Vision.Sharpness import Sharpness
from LAMDA_SSL.Augmentation.Vision.ShearX import ShearX
from LAMDA_SSL.Augmentation.Vision.ShearY import ShearY
from LAMDA_SSL.Augmentation.Vision.Solarize import Solarize
from LAMDA_SSL.Augmentation.Vision.TranslateX import TranslateX
from LAMDA_SSL.Augmentation.Vision.TranslateY import TranslateY
from LAMDA_SSL.Base.Transformer import Transformer
import numpy as np
import random

class RandAugment(Transformer):
    def __init__(self, n=2, m=5, num_bins=10, random=True,augment_list=None):
        # >> Parameter:
        # >> - n: The times of Random augmentation.
        # >> - m: The magnitude of Random augmentation.
        # >> - num_bins: The number of intervals  division for the value of the augmentation.
        # >> - random: Whether to use random value for augmentation.
        # >> - augment_list: The list of augmentations and their minimum and maximum values.
        super().__init__()
        self.n = n
        self.m = m
        self.num_bins=num_bins
        self.random=random
        self.augment_list =[(AutoContrast, None, None),
            (Brightness, 0.05, 0.95),
            (Color, 0.05, 0.95),
            (Contrast, 0.05, 0.95),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 8),
            (Rotate,  0.0,30.0),
            (Sharpness, 0.05, 0.95),
            (ShearX, 0.0, 0.3),
            (ShearY, 0.0, 0.3),
            (Solarize,  0.0,255.0),
            (TranslateX, 0.0, 0.3),
            (TranslateY, 0.0, 0.3)] if augment_list is None else augment_list

    def transform(self, X):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_v, max_v in ops:
            if max_v is None and max_v is None:
                aug=op()
            else:
                if self.random:
                    m = np.random.randint(1, self.m)
                else:
                    m=self.m
                aug=op(min_v=min_v,max_v=max_v,num_bins=self.num_bins,magnitude=m)
            if random.random() < 0.5:
                X=aug.fit_transform(X)
        return X