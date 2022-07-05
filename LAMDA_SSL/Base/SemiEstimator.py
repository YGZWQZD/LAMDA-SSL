from sklearn.base import BaseEstimator
from abc import ABC,abstractmethod
class SemiEstimator(ABC,BaseEstimator):
    @abstractmethod
    def fit(self,X,y,unlabeled_X):
        raise NotImplementedError(
            "The fit() method of SemiEstimator must be implemented."
        )