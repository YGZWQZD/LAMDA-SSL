from abc import ABC,abstractmethod

class RegressorEvaluation(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def scoring(self,y_true,y_pred=None):
        # > scoring(y_true,y_pred=None): Score the performace of the model.
        # >> - y_true: Ground-truth labels.
        # >> - y_pred: The results of model's predictions.
        raise NotImplementedError