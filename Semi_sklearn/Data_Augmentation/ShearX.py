from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
import torchvision.transforms.functional as F
import math
class ShearX(Augmentation):
    def __init__(self, v):
        super().__init__()
        self.v=v

    def transform(self,X):
        if X is not None:
            # X = to_image(X)
            X=F.affine(X,angle=0,translate=[0, 0],scale=1.0,shear=[math.degrees(self.v),0.0])
            # if y is not None:
            #     return X,y
            return X
        # elif dataset is not None:
        #     X = dataset.get_X()
        #     X= X.transform(X.size, PIL.Image.AFFINE, (1, self.v, 0, 0, 1, 0))
        #     return dataset.set_X(X)
        else:
            raise ValueError('No data to augment')