from abc import ABC,abstractmethod
class EvaluationRegressor(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def scoring(self,y_true,y_pred=None):
        raise NotImplementedError