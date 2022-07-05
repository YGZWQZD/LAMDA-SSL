from .SemiEstimator import SemiEstimator
from abc import abstractmethod

class InductiveEstimator(SemiEstimator):
    __semi_type__='Inductive'
    @abstractmethod
    def predict(self,X):
    # >> predict(X): Make predictions on the new data.
    # >> - X: Samples to be predicted.
        raise NotImplementedError(
            "Predict method must be implemented."
        )