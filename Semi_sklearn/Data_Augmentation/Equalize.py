import PIL, PIL.ImageOps
from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
class AutoContrast(Augmentation):
    def __init__(self):
        super().__init__()
    def fit(self,X=None,y=None,dataset=None):
        pass

    def transform(self,X=None,y=None,dataset=None):
        if X is not None:
            X=PIL.ImageOps.equalize(X)
            if y is not None:
                return X,y
            return X
        elif dataset is not None:
            X=PIL.ImageOps.equalize(dataset.get_X())
            return dataset.set_X(X)
        else:
            raise ValueError('No data to augment')

    def fit_transform(self,X=None,y=None,dataset=None):

        return self.transform(X=X,y=y,dataset=dataset)