from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation
import numpy as np
class Mixup(Augmentation):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.lam=None
        self.x=None
        self.y=None

    def fit(self,X,y=None,**fit_params):
        self.x,self.y=X
        return self

    def transform(self,X):
        _x,_y=X
        self.lam = np.random.beta(self.alpha, self.alpha)
        mixed_x = self.lam * self.x + (1 - self.lam) * _x
        mixed_y = self.lam * self.y + (1 - self.lam) * _y
        return mixed_x, mixed_y
