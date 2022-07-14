from LAMDA_SSL.Base.Transformer import Transformer
class MaxAbsScaler(Transformer):
    def __init__(self,max_abs=None):
        super().__init__()
        # >> Parameter:
        # >> - max_abs: The max abs value.
        self.max_abs=max_abs

    def transform(self,X):
        max_abs=X.abs().max() if self.max_abs is None else self.max_abs
        return X/max_abs