import PIL,PIL.ImageEnhance
from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
class TranslateY(Augmentation):
    def __init__(self, v):
        self.v=v

    def fit(self,X=None,y=None,dataset=None):
        pass

    def transform(self,X=None,y=None,dataset=None):

        if X is not None:
            v = self.v *X.size[1]
            X=X.transform(X.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1,v))
            X=PIL.ImageEnhance.Sharpness(X).enhance(self.v)
            if y is not None:
                return X,y
            return X
        elif dataset is not None:
            X = dataset.get_X()
            v = self.v * X.size[1]
            X= X.transform(X.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1,v))
            return dataset.set_X(X)
        else:
            raise ValueError('No data to augment')

    def fit_transform(self,X=None,y=None,dataset=None):

        return self.transform(self,X=X,y=y,dataset=dataset)