
import torch

from lamda_ssl.Transform.Transformer import Transformer
class ToTensor(Transformer):
    def __init__(self):
        super().__init__()

    def transform(self,X):
        X=torch.Tensor(X)
        return X
