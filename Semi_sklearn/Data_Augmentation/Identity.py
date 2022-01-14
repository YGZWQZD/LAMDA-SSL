from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation

class Identity(Augmentation):
    def __init__(self):
        super().__init__()

    def transform(self,X=None):
        if X is not None:
            return X
        else:
            raise ValueError('No data to augment')