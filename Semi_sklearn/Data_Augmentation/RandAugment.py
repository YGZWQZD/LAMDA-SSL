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

def augment_list():
    # FixMatch paper
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
    return augs


class RandAugment(Augmentation):
    def __init__(self, n, m):
        super().__init__()
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.parameter_max=10
        self.augment_list = augment_list()

    def _int_parameter(self,v,max_v):
        return int(v * max_v / self.parameter_max)

    def _float_parameter(self,v, max_v):
        return float(v) * max_v / self.parameter_max

    def transform(self, X):
        ops = random.choices(self.augment_list, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                if op in [AutoContrast,Equalize,Identity]:
                    aug=op()
                else:
                    if op in [Rotate,Posterize,Solarize]:
                        _v=self._int_parameter(v, max_v) + bias
                    else:
                        _v = self._float_parameter(v, max_v) + bias
                    aug=op(v=_v)
                X=aug.fit_transform(X)
        X = Cutout(0.5).fit_transform(X)
        return X