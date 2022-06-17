from .SemiEstimator import SemiEstimator
from abc import abstractmethod

class InductiveEstimator(SemiEstimator):
    __semi_type__='Inductive'
    @abstractmethod
    def predict(self,X):
        raise NotImplementedError(
            "Predict method must be implemented."
        )