import copy
import numbers
import random

import PIL.Image

from LAMDA_SSL.Base.Transformer import Transformer

import numpy as np
import torch

class CutoutAbs(Transformer):
    def __init__(self, v=16,fill=(127,127,127),random_v=True):
        # >> Parameter:
        # >> - v: The absolute value of the crop size.
        # >> - fill: The padding value.
        # >> - random_v: Whether to randomly determine the crop size.
        super().__init__()
        self.v=v
        self.random_v=random_v
        if isinstance(self.v, (tuple, list)):
            self.vx, self.vy = self.v[0], self.v[1]
        else:
            self.vx, self.vy = self.v, self.v

        if isinstance(fill,numbers.Number):
            self.fill = (fill, fill, fill)
        else:
            self.fill=fill

    def transform(self,X):
        if isinstance(X,np.ndarray):
            X=PIL.Image.fromarray(X)
        if isinstance(X,PIL.Image.Image):
            w, h = X.size[0], X.size[1]
            x0 = np.random.uniform(w)
            y0 = np.random.uniform(h)
            rd=random.random() if self.random_v else 1
            vx=self.vx *rd
            vy = self.vy * rd
            x0 = int(max(0, x0 - vx / 2.))
            y0 = int(max(0, y0 - vy / 2.))
            x1 = int(min(w, x0 + vx))
            y1 = int(min(h, y0 + vy))
            xy = (x0, y0, x1, y1)
            PIL.ImageDraw.Draw(X).rectangle(xy, self.fill)
            return X

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
                    rd = random.random() if self.random_v else 1
                    vx = self.vx * rd
                    vy = self.vy * rd
                    x0 = int(max(0, x0 - vx / 2.))
                    y0 = int(max(0, y0 - vy  / 2.))
                    x1 = int(min(w, x0 + vx ))
                    y1 = int(min(h, y0 + vy ))

                    X[_, 0, x0:x1, y0:y1].copy_(torch.Tensor([self.fill[0]]))
                    X[_, 1, x0:x1, y0:y1].copy_(torch.Tensor([self.fill[1]]))
                    X[_, 2, x0:x1, y0:y1].copy_(torch.Tensor([self.fill[2]]))
            else:
                x0 = np.random.uniform(w)
                y0 = np.random.uniform(h)
                rd = random.random() if self.random_v else 1
                vx = self.vx * rd
                vy = self.vy * rd
                x0 = int(max(0, x0 - vx / 2.))
                y0 = int(max(0, y0 - vy / 2.))
                x1 = int(min(w, x0 + vx))
                y1 = int(min(h, y0 + vy))
                # print('???')
                X[0,x0:x1,y0:y1].copy_(torch.Tensor([self.fill[0]]))
                X[1,x0:x1,y0:y1].copy_(torch.Tensor([self.fill[1]]))
                X[2,x0:x1,y0:y1].copy_(torch.Tensor([self.fill[2]]))

            return X
        else:
            raise ValueError('No data to augment')