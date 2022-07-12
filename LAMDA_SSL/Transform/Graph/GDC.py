from LAMDA_SSL.Base.Transformer import Transformer
import torch_geometric.transforms as gt
class GDC(Transformer):
    def __init__(self,self_loop_weight=1, normalization_in='sym',
                 normalization_out='col',
                 diffusion_kwargs=dict(method='ppr', alpha=0.15),
                 sparsification_kwargs=dict(method='threshold',avg_degree=64),
                 exact=True):
        # >> Parameter:
        # >> - self_loop_weight: Weight of the added self-loop. Set to None to add no self-loops.
        # >> - normalization_in: Normalization of the transition matrix on the original (input) graph. Possible values: "sym", "col", and "row"`.
        # >> - normalization_out: Normalization of the transition matrix on the transformed GDC (output) graph. Possible values: "sym", "col", and "row"`.
        # >> - diffusion_kwargs: Dictionary containing the parameters for diffusion.
        # >> - sparsification_kwargs: Dictionary containing the parameters for sparsification.
        # >> - exact: Whether to accurately calculate the diffusion matrix.
        super().__init__()
        self.gdc=gt.GDC(self_loop_weight=self_loop_weight,normalization_in=normalization_in,
                        normalization_out=normalization_out,diffusion_kwargs=diffusion_kwargs,
                        sparsification_kwargs=sparsification_kwargs,exact=exact)

    def transform(self,X):
        X=self.gdc(X)
        return X
