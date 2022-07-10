from LAMDA_SSL.Transform.Transformer import Transformer
import torchvision.transforms.functional as F
import PIL
import numpy as np
from torchvision.transforms import InterpolationMode
from LAMDA_SSL.utils import partial
class Resize(Transformer):
    def __init__(self,size, interpolation = InterpolationMode.BILINEAR,
           max_size = None, antialias = None):
        super().__init__()
        self.resize=partial(F.resize,size=size,interpolation=interpolation,max_size=max_size,antialias=antialias)

    def transform(self,X):
        if isinstance(X,np.ndarray):
            X=PIL.Image.fromarray(X)
        X = self.resize(X)
        return X