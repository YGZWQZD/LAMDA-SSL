from sklearn.base import BaseEstimator
from abc import ABC,abstractmethod

class SemiEstimator(ABC,BaseEstimator):
    @abstractmethod
    def fit(self,X,y,unlabeled_X):
        raise NotImplementedError(
            "Fit method of SemiEstimator class must be implemented."
        )