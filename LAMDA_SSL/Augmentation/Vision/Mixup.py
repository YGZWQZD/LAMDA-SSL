from LAMDA_SSL.Base.Transformer import Transformer
import numpy as np

class Mixup(Transformer):
    def __init__(self, alpha=0.5):
        # >> Parameter:
        # >> - alpha: The parameter of the beta distribution.
        super().__init__()
        self.alpha = alpha
        self.lam=None
        self.X=None

    def fit(self,X,y=None,**fit_params):
        self.X=X
        self.y=y
        return self

    def transform(self,X):
        self.lam = np.random.beta(self.alpha, self.alpha)
        if self.y is not None and isinstance(X,(list,tuple)):
            X,y=X[0],X[1]
            X = self.lam * self.X + (1 - self.lam) * X
            y = self.lam * self.y + (1 - self.lam) * y
            return X,y
        else:
            if isinstance(X,(list,tuple)):
                X=type(X)(self.lam * self.X[i]+(1 - self.lam) * X[i] for i in range(len(X)))
            else:
                X=self.lam * self.X+(1 - self.lam) * X
            return X
