from torchtext import vocab
from Semi_sklearn.Transform.Transformer import Transformer
class Vectors(Transformer):
    def __init__(self,name, cache=None, url=None, unk_init=None, max_vectors=None):
        super(Vectors, self).__init__()
        self.vec=vocab.Vectors(name,cache,url,unk_init,max_vectors)

    def transform(self,X):
        if isinstance(X,tuple):
            X=list(X)
        if isinstance(X,list):
            return self.vec.get_vecs_by_tokens(X)
        else:
            return self.vec[X]
