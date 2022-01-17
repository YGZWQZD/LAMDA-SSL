from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
import torchvision.transforms.functional as F
class TranslateY(Augmentation):
    def __init__(self, v):
        super().__init__()
        self.v=v

    def transform(self,X):
        if X is not None:
            # X = to_image(X)
            X = F.affine(X, angle=0, translate=[ 0,self.v], scale=1.0, shear=[0.0, 0.0])
            return X
        else:
            raise ValueError('No data to augment')