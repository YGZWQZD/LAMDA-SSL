from .SemiEstimator import SemiEstimator
from abc import abstractmethod

class TransductiveEstimator(SemiEstimator):
    __semi_type__='Transductive'
    @abstractmethod
    def predict(self,X=None,Transductive=True):
        raise NotImplementedError(
            "Predict method must be implemented."
        )