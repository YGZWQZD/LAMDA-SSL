from LAMDA_SSL.Base.Transformer import Transformer
import torch_geometric.transforms as gt
class GCNNorm(Transformer):
    def __init__(self,add_self_loops=True):
        # >> Parameter:
        # >> - add_self_loops: Whether to add self loops.
        super().__init__()
        self.GCNNorm=gt.GCNNorm(add_self_loops=add_self_loops)

    def transform(self,X):
        X=self.GCNNorm(X)
        return X