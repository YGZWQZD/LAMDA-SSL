from Semi_sklearn.Transform.Normalization import Normalization
from Semi_sklearn.Transform.ImageToTensor import ToTensor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from Semi_sklearn.Transform.ToImage import ToImage
class VisionMixin:
    def __init__(self):
        pass

    def init_transforms(self):
        self.transforms=None
        self.target_transform=None
        self.pre_transform=ToImage()
        self.transform=Pipeline([('ToTensor',ToTensor()),
                              # ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        self.unlabeled_transform=Pipeline([('ToTensor',ToTensor()),
                              # ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        self.test_transform=Pipeline([('ToTensor',ToTensor()),
                              # ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        self.valid_transform=Pipeline([('ToTensor',ToTensor()),
                              # ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        return self


    def show_image(self,img):
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return img