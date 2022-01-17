from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
from torchvision import transforms

class RandomHorizontalFlip(Augmentation):
    def __init__(self):
        super().__init__()
        self.augumentation=transforms.RandomHorizontalFlip()

    def transform(self,X=None):
        if X is not None:
            X=self.augumentation(X)
            return X
        else:
            raise ValueError('No data to augment')
