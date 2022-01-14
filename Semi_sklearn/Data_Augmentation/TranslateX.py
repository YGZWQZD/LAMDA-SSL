from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
import torchvision.transforms.functional as F
class TranslateX(Augmentation):
    def __init__(self, v):
        super().__init__()
        self.v=v

    def transform(self,X):
        if X is not None:
            X = F.affine(X, angle=0, translate=[self.v, 0], scale=1.0, shear=[0.0, 0.0])
            return X
        else:
            raise ValueError('No data to augment')