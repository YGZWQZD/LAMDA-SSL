import PIL,PIL.ImageEnhance
from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
class TranslateX(Augmentation):
    def __init__(self, v):
        super().__init__()
        self.v=v

    def fit(self,X=None,y=None,dataset=None):
        pass

    def transform(self,X=None,y=None,dataset=None):

        if X is not None:
            v = self.v *X.size[0]
            X=X.transform(X.size, PIL.Image.AFFINE, (1, 0,v, 0, 1, 0))
            X=PIL.ImageEnhance.Sharpness(X).enhance(self.v)
            if y is not None:
                return X,y
            return X
        elif dataset is not None:
            X = dataset.get_X()
            v = self.v * X.size[0]
            X= X.transform(X.size, PIL.Image.AFFINE, (1,  0,v, 0, 1, 0))
            return dataset.set_X(X)
        else:
            raise ValueError('No data to augment')

    def fit_transform(self,X=None,y=None,dataset=None):

        return self.transform(X=X,y=y,dataset=dataset)