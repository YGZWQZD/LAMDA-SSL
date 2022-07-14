import torch
import PIL
import numpy as np
from LAMDA_SSL.Base.Transformer import Transformer
import torchvision.transforms as tvt
class ToTensor(Transformer):
    def __init__(self,dtype=None,image=False):
        # > - Parameter:
        # >> - dtype: The dtype of Tensor.
        # >> - image: Whether the X is a image.
        super().__init__()
        self.dtype=dtype
        self.image=image

    def transform(self,X):
        if self.image is True:
            if isinstance(X, np.ndarray):
                X = PIL.Image.fromarray(X)
            if isinstance(X,PIL.Image.Image):
                X=tvt.ToTensor()(X)
            if self.dtype == 'float' or self.dtype=='float32':
                X=X.float()
            elif self.dtype == 'double' or self.dtype == 'float64':
                X=X.double()
            elif self.dtype == 'uint8' or self.dtype == 'uint' or self.dtype == 'byte':
                X=X.byte()
            elif self.dtype == 'int8' or self.dtype == 'char':
                X =X.char()
            elif self.dtype == 'int16' or self.dtype == 'short':
                X =X.short()
            elif self.dtype == 'int32' or self.dtype == 'int':
                X =X.int()
            elif self.dtype == 'int64' or self.dtype == 'long':
                X =X.long()
        else:
            if self.dtype == 'float' or self.dtype=='float32':
                X=torch.FloatTensor(X)
            elif self.dtype == 'double' or self.dtype=='float64':
                X=torch.DoubleTensor(X)
            elif self.dtype == 'uint8' or self.dtype=='uint' or self.dtype=='byte':
                X=torch.ByteTensor(X)
            elif self.dtype == 'int8' or self.dtype=='char':
                X=torch.CharTensor(X)
            elif self.dtype == 'int16'or self.dtype=='short':
                X = torch.ShortTensor(X)
            elif self.dtype == 'int32'or self.dtype=='int':
                X = torch.IntTensor(X)
            elif self.dtype == 'int64'or self.dtype=='long':
                X = torch.LongTensor(X)
            else:
                X=torch.Tensor(X)
        return X
