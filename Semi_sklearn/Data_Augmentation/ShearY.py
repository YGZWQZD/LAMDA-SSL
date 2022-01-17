from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
import torchvision.transforms.functional as F
import math

class ShearY(Augmentation):
    def __init__(self, v):
        super().__init__()
        self.v=v

    def transform(self,X):
        if X is not None:
            X=F.affine(X,angle=0,translate=[0, 0],scale=1.0,shear=[0.0,math.degrees(self.v)])
            return X
        # elif dataset is not None:
        #     X = dataset.get_X()
        #     X= X.transform(X.size, PIL.Image.AFFINE, (1,  0, 0,self.v, 1, 0))
        #     return dataset.set_X(X)
        else:
            raise ValueError('No data to augment')
