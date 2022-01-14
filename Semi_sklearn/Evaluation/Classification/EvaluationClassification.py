from abc import ABC,abstractmethod
class EvaluationClassification(ABC):
    def __init__(self):
        pass
    def scoring(self,y_pred,y_est,y_lable):
        raise NotImplementedError