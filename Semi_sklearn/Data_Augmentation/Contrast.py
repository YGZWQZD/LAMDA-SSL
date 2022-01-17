from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
import torchvision.transforms.functional as F
class Contrast(Augmentation):
    def __init__(self, v):
        super().__init__()
        self.v=v
        assert v >= 0.0

    def transform(self,X):
        if X is not None:
            X=F.adjust_contrast(X,1.0+self.v)
            return X
        else:
            raise ValueError('No data to augment')