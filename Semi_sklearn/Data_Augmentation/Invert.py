import PIL, PIL.ImageOps
from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
import torchvision.transforms.functional as F
class Invert(Augmentation):

    def __init__(self):
        super().__init__()

    def transform(self,X):
        if X is not None:
            X=F.invert(X)
            return X
        else:
            raise ValueError('No data to augment')