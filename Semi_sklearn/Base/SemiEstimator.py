from sklearn.base import BaseEstimator,ClassifierMixin
from abc import ABC,abstractmethod

class SemiEstimator(ABC,BaseEstimator):
    @abstractmethod
    def fit(self,X,y,unlabled_X,**params):
        raise NotImplementedError(
            "Fit method of SemiEstimator class must be implemented."
        )
