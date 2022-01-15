from abc import ABC,abstractmethod
class EvaluationClassification(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def scoring(self,y_true,y_pred=None,y_score=None):
        raise NotImplementedError