from torchtext import vocab
from Semi_sklearn.Transform.Transformer import Transformer
class FastText(Transformer):
    def __init__(self, language="en"):
        super().__init__()
        self.vec=vocab.FastText(language=language)

    def transform(self,X):
        if isinstance(X,tuple):
            X=list(X)
        if isinstance(X,list):
            return self.vec.get_vecs_by_tokens(X)
        else:
            return self.vec[X]
