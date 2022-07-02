from lamda_ssl.Transform.AutoContrast import AutoContrast
from lamda_ssl.Transform.Brightness import Brightness
from lamda_ssl.Transform.Color import Color
from lamda_ssl.Transform.Contrast import Contrast
from lamda_ssl.Transform.Equalize import Equalize
from lamda_ssl.Transform.Identity import Identity
from lamda_ssl.Transform.Posterize import Posterize
from lamda_ssl.Transform.Rotate import Rotate
from lamda_ssl.Transform.Sharpness import Sharpness
from lamda_ssl.Transform.ShearX import ShearX
from lamda_ssl.Transform.ShearY import ShearY
from lamda_ssl.Transform.Solarize import Solarize
from lamda_ssl.Transform.TranslateX import TranslateX
from lamda_ssl.Transform.TranslateY import TranslateY
from lamda_ssl.Transform.Transformer import Transformer
import random

class RandAugment(Transformer):
    def __init__(self, n=2, m=5, num_bins=10, random=True,augment_list=None):
        super().__init__()
        self.n = n
        self.m = m
        self.num_bins=num_bins
        self.random=random
        self.augment_list =[[(AutoContrast, None, None),
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
            (TranslateY, 0.0, 0.3)]] if augment_list is None else augment_list

    def transform(self, X):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_v, max_v in ops:
            if max_v is None and max_v is None:
                aug=op()
            else:
                if self.random:
                    m=random.choice(range(1,self.m+1))
                else:
                    m=self.m
                aug=op(min_v=min_v,max_v=max_v,num_bins=self.num_bins,magnitude=m)
            if random.random() < 0.5:
                X=aug.fit_transform(X)
        return X