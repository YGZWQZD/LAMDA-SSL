from Semi_sklearn.Transform.Normalization import Normalization
from Semi_sklearn.Transform.ToTensor import ToTensor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
class VisionMixin:
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std

    def init_transforms(self):
        self.transforms=None
        self.target_transform=None
        self.transform=Pipeline([('ToTensor',ToTensor()),
                              ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        self.unlabled_transform=Pipeline([('ToTensor',ToTensor()),
                              ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        self.test_transform=Pipeline([('ToTensor',ToTensor()),
                              ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        self.valid_transform=Pipeline([('ToTensor',ToTensor()),
                              ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        return self


    def show_image(self,img):
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return img







# class AudioMixin:
#     #听语音：
#     #train 得到 logits ，标记数据和 text（lable）计算损失，无标记数据计算一致性损失
#     #default transforms
