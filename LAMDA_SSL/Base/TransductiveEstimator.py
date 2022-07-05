from .SemiEstimator import SemiEstimator
from abc import abstractmethod

class TransductiveEstimator(SemiEstimator):
    __semi_type__='Transductive'
    @abstractmethod
    def predict(self,X=None,Transductive=True):
    # >> predict(X=None,Transductive=True): Output the result of transductive learning or make predictions on the new data.
    # >> - X: The samples to be predicted. It is only valid when Transductive is False.
    # >> - Transductive: Whether to use transductive learning mechanism to directly output the prediction result of unlabeled_X input during fit.
        raise NotImplementedError(
            "Predict method must be implemented."
        )