from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
import torchvision.transforms.functional as F
class Solarize(Augmentation):
    def __init__(self, v):
        super().__init__()
        self.v=v
        assert 0 <= v <= 256

    def transform(self,X):
        if X is not None:
            X=F.solarize(X, self.v)
            return X
        else:
            raise ValueError('No data to augment')