import PIL,PIL.ImageEnhance
from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
import numpy as np
def CutoutAbs(X, v):
    if v < 0:
        return X
    w, h = X.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    X = X.copy()
    PIL.ImageDraw.Draw(X).rectangle(xy, color)
    return X
class Cutout(Augmentation):
    def __init__(self, v):
        super().__init__()
        self.v=v
        assert 0.0 <= v <= 0.5
    def fit(self,X=None,y=None,dataset=None):
        pass

    def transform(self,X=None,y=None,dataset=None):
        if X is not None:
            v = self.v * X.size[0]
            X=CutoutAbs(X, v)
            if y is not None:
                return X,y
            return X
        elif dataset is not None:
            X=dataset.get_X()
            v = self.v * X.size[0]
            X=CutoutAbs(X, v)
            return dataset.set_X(X)
        else:
            raise ValueError('No data to augment')

    def fit_transform(self,X=None,y=None,dataset=None):

        return self.transform(X=X,y=y,dataset=dataset)