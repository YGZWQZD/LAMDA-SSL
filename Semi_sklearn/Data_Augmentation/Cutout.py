import PIL,PIL.ImageEnhance
from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
from Semi_sklearn.Data_Augmentation.CutoutAbs import CutoutAbs
class Cutout(Augmentation):
    def __init__(self, v):
        super().__init__()
        self.v=v
        assert 0.0 <= v <= 0.5

    def transform(self,X):
        if X is not None:
            v=(self.v*X.shape[-2],self.v*X.shape[-1])
            X=CutoutAbs(v).fit_transform(X)
            return X
        else:
            raise ValueError('No data to augment')
