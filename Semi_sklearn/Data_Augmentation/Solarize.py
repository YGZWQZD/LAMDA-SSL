import PIL, PIL.ImageOps
from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
class Solarize(Augmentation):
    def __init__(self, v):
        super().__init__()
        self.v=v
        assert 0 <= v <= 256
    def fit(self,X=None,y=None,dataset=None):
        pass

    def transform(self,X=None,y=None,dataset=None):
        if X is not None:
            X=PIL.ImageOps.solarize(X, self.v)
            if y is not None:
                return X,y
            return X
        elif dataset is not None:
            X = PIL.ImageOps.solarize(dataset.get_X(), self.v)
            return dataset.set_X(X)
        else:
            raise ValueError('No data to augment')

    def fit_transform(self,X=None,y=None,dataset=None):

        return self.transform(X=X,y=y,dataset=dataset)