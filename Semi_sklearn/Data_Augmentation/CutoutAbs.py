import copy
import copyreg
import numbers

import PIL.Image

from Semi_sklearn.Data_Augmentation.Augmentation import Augmentation

import numpy as np
import torch

class CutoutAbs(Augmentation):
    def __init__(self, v,fill):
        super().__init__()
        self.v=v
        if isinstance(self.v, (tuple, list)):
            self.vx, self.vy = self.v[0], self.v[1]
        else:
            self.vx, self.vy = self.v, self.v

        if isinstance(fill,numbers.Number):
            self.fill = (fill, fill, fill)
        else:
            self.fill=fill

    def transform(self,X):

        if isinstance(X,PIL.Image.Image):
            X = X.copy()
            w, h = X.size[0], X.size[1]
            x0 = np.random.uniform(w)
            y0 = np.random.uniform(h)
            x0 = int(max(0, x0 - self.vx*w / 2.))
            y0 = int(max(0, y0 - self.vy*h / 2.))
            x1 = int(min(w, x0 + self.vx*w))
            y1 = int(min(h, y0 + self.vy*h))

            xy = (x0, y0, x1, y1)

            PIL.ImageDraw.Draw(X).rectangle(xy, self.fill)

        elif isinstance(X,torch.Tensor):
            X=copy.copy(X)
            X_shape=X.shape
            if self.fill is None:
                self.fill=(0,0,0)
            w, h = X_shape[-2],X_shape[-1]
            if len(X_shape) == 4:

                for _ in range(X.shape[0]):
                    x0 = np.random.uniform(w)
                    y0 = np.random.uniform(h)
                    x0 = int(max(0, x0 - self.vx * w / 2.))
                    y0 = int(max(0, y0 - self.vy * h / 2.))
                    x1 = int(min(w, x0 + self.vx * w))
                    y1 = int(min(h, y0 + self.vy * h))

                    X[_, 0, x0:x1, y0:y1].copy_(torch.Tensor([self.fill[0]]))
                    X[_, 1, x0:x1, y0:y1].copy_(torch.Tensor([self.fill[1]]))
                    X[_, 2, x0:x1, y0:y1].copy_(torch.Tensor([self.fill[2]]))
            else:
                x0 = np.random.uniform(w)
                y0 = np.random.uniform(h)
                x0 = int(max(0, x0 - self.vx * w / 2.))
                y0 = int(max(0, y0 - self.vy * h / 2.))
                x1 = int(min(w, x0 + self.vx * w))
                y1 = int(min(h, y0 + self.vy * h))
                X[0,x0:x1,y0:y1].copy_(torch.Tensor([self.fill[0]]))
                X[1,x0:x1,y0:y1].copy_(torch.Tensor([self.fill[1]]))
                X[2,x0:x1,y0:y1].copy_(torch.Tensor([self.fill[2]]))

            return X
        else:
            raise ValueError('No data to augment')