from LAMDA_SSL.Base.Transformer import Transformer
import torch_geometric.transforms as gt

class NormalizeFeatures(Transformer):
    def __init__(self,attrs=["x"]):
        # >> Parameter:
        # >> - attrs: Properties that require regularization.
        super().__init__()
        self.attrs=attrs
        self.normalize=gt.NormalizeFeatures(attrs)


    def transform(self,X):
        if X is not None:
            X=self.normalize(X)
            return X
        else:
            raise ValueError('No data to augment')