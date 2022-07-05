from LAMDA_SSL.Transform.Transformer import Transformer
import torch_geometric.transforms as gt
class GDC(Transformer):
    def __init__(self,self_loop_weight=1, normalization_in='sym',
                 normalization_out='col',
                 diffusion_kwargs=dict(method='ppr', alpha=0.15),
                 sparsification_kwargs=dict(method='threshold',avg_degree=64),
                 exact=True):
        super().__init__()
        self.gdc=gt.GDC(self_loop_weight=self_loop_weight,normalization_in=normalization_in,
                        normalization_out=normalization_out,diffusion_kwargs=diffusion_kwargs,
                        sparsification_kwargs=sparsification_kwargs,exact=exact)

    def transform(self,X):
        if X is not None:
            X=self.gdc(X)
            return X
        else:
            raise ValueError('No data to augment')