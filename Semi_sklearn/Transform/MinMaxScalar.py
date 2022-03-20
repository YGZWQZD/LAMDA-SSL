import torch

from Semi_sklearn.Transform.Transformer import Transformer
class MinMaxScalar(Transformer):
    def __init__(self,min_val=None,max_val=None):
        super().__init__()
        self.min_val=min_val
        self.max_val=max_val
    def transform(self,X):
        min_val=X.min() if self.min_val is None else self.min_val
        max_val = X.max() if self.max_val is None else self.max_val
        return (X-min_val)/(max_val-min_val)