from LAMDA_SSL.Base.Transformer import Transformer

class Identity(Transformer):
    def __init__(self):
        super().__init__()

    def transform(self,X):
        if X is not None:
            return X
        else:
            raise ValueError('No data to augment')