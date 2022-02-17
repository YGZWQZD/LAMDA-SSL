from torchtext import vocab
from Semi_sklearn.Transform.Transformer import Transformer
class CharNGram(Transformer):
    def __init__(self):
        super().__init__()
        self.vec=vocab.CharNGram()

    def transform(self,X):
        if isinstance(X,tuple):
            X=list(X)
        if isinstance(X,list):
            return self.vec.get_vecs_by_tokens(X)
        else:
            return self.vec[X]
