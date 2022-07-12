from LAMDA_SSL.Base.Transformer import Transformer
import torch_geometric.transforms as gt
class SVDFeatureReduction(Transformer):
    def __init__(self,out_channels):
        # >> Parameter:
        # >> - out_channels: The dimensionlity of node features after reduction.
        super().__init__()
        self.svd_feature_reduction=gt.SVDFeatureReduction(out_channels=out_channels)

    def transform(self,X):
        X=self.svd_feature_reduction(X)
        return X
