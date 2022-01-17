from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation

import numpy as np
import torch

class CutoutAbs(Augmentation):
    def __init__(self, v):
        super().__init__()
        self.v=v

    def transform(self,X):
        if X is not None:
            X_shape=X.shape
            if isinstance(self.v,(tuple,list)):
                vx,vy=self.v[0],self.v[1]
            else:
                vx,vy=self.v,self.v
            w, h = X.shape[-2],X.shape[-1]
            x0 = np.random.uniform(w)
            y0 = np.random.uniform(h)
            x0 = int(max(0, x0 - vx*w / 2.))
            y0 = int(max(0, y0 - vy*h / 2.))
            x1 = int(min(w, x0 + vx*w))
            y1 = int(min(h, y0 + vy*h))
            xy = (x0, y0, x1, y1)
            color = (125, 123, 114)
            if len(X_shape)==3:
                X[0,x0:x1,y0:y1].copy_(torch.Tensor([color[0]]))
                X[1,x0:x1,y0:y1].copy_(torch.Tensor([color[1]]))
                X[2,x0:x1,y0:y1].copy_(torch.Tensor([color[2]]))
            elif len(X_shape)==4:
                X[:,0, x0:x1, y0:y1].copy_(torch.Tensor([color[0]]))
                X[:,1, x0:x1, y0:y1].copy_(torch.Tensor([color[1]]))
                X[:,2, x0:x1, y0:y1].copy_(torch.Tensor([color[2]]))
            return X
        else:
            raise ValueError('No data to augment')