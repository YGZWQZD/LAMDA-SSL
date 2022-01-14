from .SemiEstimator import SemiEstimator
from abc import abstractmethod

class InductiveEstimator(SemiEstimator):
    _semi_type='Inductive'
    @abstractmethod
    def predict(self,X):
        raise NotImplementedError(
            "Predict method must be implemented."
        )