from sklearn.base import BaseEstimator
from abc import ABC,abstractmethod
class SemiEstimator(ABC,BaseEstimator):
    @abstractmethod
    def fit(self,X,y,unlabeled_X):
    # >> fit(X,y,unlabeled_X): Train a SSL model.
    # >> - X: Instances of labeled data.
    # >> - y: Labels of labeled data.
    # >> - unlabeled_X: Instances of unlabeled data.
        raise NotImplementedError(
            "The fit() method of SemiEstimator must be implemented."
        )