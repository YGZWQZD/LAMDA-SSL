from abc import abstractmethod,ABC
from sklearn.base import BaseEstimator, TransformerMixin

class Transformer(BaseEstimator,TransformerMixin,ABC):
    def __init__(self):
        pass
    def fit(self,X,y=None,**fit_params):
        return self

    @abstractmethod
    def transform(self,X):
        raise NotImplementedError('Transform method of Augmentation class must be implemented.')

    def fit_transform(self,X,y=None,**fit_params):
        return self.fit(X=X,y=y,fit_params=fit_params).transform(X)
