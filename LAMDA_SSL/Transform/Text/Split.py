from LAMDA_SSL.Base.Transformer import Transformer

class Split(Transformer):
    def __init__(self):
        super().__init__()

    def transform(self,X):
        X = X.split()
        return X
