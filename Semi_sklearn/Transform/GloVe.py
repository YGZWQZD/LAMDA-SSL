from torchtext import vocab
from Semi_sklearn.Transform.Transformer import Transformer
class Glove(Transformer):
    def __init__(self,name="840B", dim=300):
        super().__init__()
        self.vec=vocab.GloVe(name,dim)

    def transform(self,X):
        if isinstance(X,tuple):
            X=list(X)
        if isinstance(X,list):
            return self.vec.get_vecs_by_tokens(X)
        else:
            return self.vec[X]
