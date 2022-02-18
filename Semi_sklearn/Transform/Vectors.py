from torchtext import vocab
from Semi_sklearn.Transform.Transformer import Transformer
import torch
class Vectors(Transformer):
    def __init__(self,name, cache=None, url=None, unk_init=None, max_vectors=None,lower_case_backup=True):
        super(Vectors, self).__init__()
        self.vec=vocab.Vectors(name,cache,url,unk_init,max_vectors)
        self.lower_case_backup=lower_case_backup

    def transform(self,X):
        if isinstance(X,tuple):
            X=list(X)
        if isinstance(X,list):
            if isinstance(X[0],str):
                return self.vec.get_vecs_by_tokens(X,lower_case_backup=True)
            else:
                indices = [self.vec.vectors[idx] for idx in X]
                return torch.stack(indices)
        else:
            if isinstance(X[0],str):
                return self.vec[X]
            else:
                return self.vec.vectors[X]
