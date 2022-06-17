from Semi_sklearn.Transform.AutoContrast import AutoContrast
from Semi_sklearn.Transform.Brightness import Brightness
from Semi_sklearn.Transform.Color import Color
from Semi_sklearn.Transform.Contrast import Contrast
from Semi_sklearn.Transform.Equalize import Equalize
from Semi_sklearn.Transform.Identity import Identity
from Semi_sklearn.Transform.Posterize import Posterize
from Semi_sklearn.Transform.Rotate import Rotate
from Semi_sklearn.Transform.Sharpness import Sharpness
from Semi_sklearn.Transform.ShearX import ShearX
from Semi_sklearn.Transform.ShearY import ShearY
from Semi_sklearn.Transform.Solarize import Solarize
from Semi_sklearn.Transform.TranslateX import TranslateX
from Semi_sklearn.Transform.TranslateY import TranslateY
from Semi_sklearn.Transform.Transformer import Transformer
import random



# augs = [(AutoContrast, None, None),
#             (Brightness, 0.9, 0.05),
#             (Color, 0.9, 0.05),
#             (Contrast, 0.9, 0.05),
#             (Equalize, None, None),
#             (Identity, None, None),
#             (Posterize, 4, 4),
#             (Rotate, 30, 0),
#             (Sharpness, 0.9, 0.05),
#             (ShearX, 0.3, 0),
#             (ShearY, 0.3, 0),
#             (Solarize, 256, 0),
#             (TranslateX, 0.3, 0),
#             (TranslateY, 0.3, 0)]



class RandAugment(Transformer):
    def __init__(self, n, m, num_bins,random=False,augment_list=None):
        super().__init__()
        self.n = n
        self.m = m
        self.num_bins=num_bins
        self.random=random
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
                if self.random:
                    m=random.choice(range(1,self.m+1))
                else:
                    m=self.m
                aug=op(min_v=min_v,max_v=max_v,num_bins=self.num_bins,magnitude=m)
            # print(X)
            # print(aug)
            X=aug.fit_transform(X)
        return X