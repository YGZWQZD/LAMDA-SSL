from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
from torchvision import transforms

class RandomCrop(Augmentation):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        self.augumentation=transforms.RandomCrop(size=size,padding=padding,
                                                 pad_if_needed=pad_if_needed,
                                                 fill=fill,padding_mode=padding_mode)
    def transform(self,X):
        if X is not None:
            X=self.augumentation(X)
            return X
        else:
            raise ValueError('No data to augment')
