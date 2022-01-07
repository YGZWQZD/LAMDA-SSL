import PIL,PIL.ImageEnhance
from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
class Color(Augmentation):
    def __init__(self, v):
        self.v=v
        assert v >= 0.0
    def fit(self,X=None,y=None,dataset=None):
        pass

    def transform(self,X=None,y=None,dataset=None):
        if X is not None:
            X=PIL.ImageEnhance.Color(X).enhance(self.v)
            if y is not None:
                return X,y
            return X
        elif dataset is not None:
            X = PIL.ImageEnhance.Color(dataset.get_X()).enhance(self.v)
            return dataset.set_X(X)
        else:
            raise ValueError('No data to augment')

    def fit_transform(self,X=None,y=None,dataset=None):

        return self.transform(self,X=X,y=y,dataset=dataset)