from LAMDA_SSL.Base.Transformer import Transformer
class StandardScaler(Transformer):
    def __init__(self,mean=None,std=None):
        super().__init__()
        # >> Parameter:
        # >> - mean: The value of mean.
        # >> - std: The value of standard deviation.
        self.mean=mean
        self.std=std

    def transform(self,X):
        mean=X.mean() if self.mean is None else self.mean
        std = X.std() if self.std is None else self.std
        return (X-mean)/std
