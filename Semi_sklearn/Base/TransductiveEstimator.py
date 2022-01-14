from .SemiEstimator import SemiEstimator
from abc import abstractmethod

class TransductiveEstimator(SemiEstimator):
    _semi_type='Transductive'
    transduction=None
    @abstractmethod
    def predict(self,X=None,Transductive=True,base_estimator=None):
        raise NotImplementedError(
            "Predict method must be implemented."
        )