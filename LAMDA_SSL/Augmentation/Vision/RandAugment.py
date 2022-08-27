from LAMDA_SSL.Base.Transformer import Transformer
import numpy as np
import random
import PIL

def AutoContrast(X, **kwarg):
    return PIL.ImageOps.autocontrast(X)


def Brightness(X, min_v, max_v,magnitude,num_bins=10):
    v = min_v+float(max_v -min_v) * magnitude/ num_bins
    return PIL.ImageEnhance.Brightness(X).enhance(v)


def Color(X, min_v, max_v,magnitude,num_bins=10):
    v = min_v+float(max_v -min_v) * magnitude/ num_bins
    return PIL.ImageEnhance.Color(X).enhance(v)


def Contrast(X, min_v, max_v,magnitude,num_bins=10):
    v = min_v+float(max_v -min_v) * magnitude/ num_bins
    return PIL.ImageEnhance.Contrast(X).enhance(v)

def Equalize(X, **kwarg):
    return PIL.ImageOps.equalize(X)


def Identity(X, **kwarg):
    return X


def Invert(X, **kwarg):
    return PIL.ImageOps.invert(X)


def Posterize(X, min_v, max_v,magnitude,num_bins=10):
    v = int(min_v+(max_v -min_v) * magnitude/ num_bins)
    return PIL.ImageOps.posterize(X, v)


def Rotate(X, min_v, max_v,magnitude,num_bins=10):
    v = int(min_v+(max_v -min_v) * magnitude/ num_bins)
    if random.random() < 0.5:
        v = -v
    return X.rotate(v)


def Sharpness(X, min_v, max_v,magnitude,num_bins=10):
    v = min_v+float(max_v -min_v) * magnitude/ num_bins
    return PIL.ImageEnhance.Sharpness(X).enhance(v)


def ShearX(X, min_v, max_v,magnitude,num_bins=10):
    v = min_v+float(max_v -min_v) * magnitude/ num_bins
    if random.random() < 0.5:
        v = -v
    return X.transform(X.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(X, min_v, max_v,magnitude,num_bins=10):
    v = min_v+float(max_v -min_v) * magnitude/ num_bins
    if random.random() < 0.5:
        v = -v
    return X.transform(X.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(X, min_v, max_v,magnitude,num_bins=10):
    v = int(min_v+(max_v -min_v) * magnitude/ num_bins)
    return PIL.ImageOps.solarize(X, 256 - v)



def TranslateX(X, min_v, max_v,magnitude,num_bins=10):
    v = min_v+float(max_v -min_v) * magnitude/ num_bins
    if random.random() < 0.5:
        v = -v
    v = int(v * X.size[0])
    return X.transform(X.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(X, min_v, max_v,magnitude,num_bins=10):
    v = min_v+float(max_v -min_v) * magnitude/ num_bins
    if random.random() < 0.5:
        v = -v
    v = int(v * X.size[1])
    return X.transform(X.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

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
            (Solarize,  0.0,256.0),
            (TranslateX, 0.0, 0.3),
            (TranslateY, 0.0, 0.3)] if augment_list is None else augment_list

    def transform(self, X):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_v, max_v in ops:
            if self.random:
                m = np.random.randint(1, self.m)
            else:
                m=self.m
            if random.random() < 0.5:
                X=op(X=X,min_v=min_v,max_v=max_v,num_bins=self.num_bins,magnitude=m)
        return X