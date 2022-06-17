from abc import ABC,abstractmethod
class EvaluationCluster(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def scoring(self,y_true=None,clusters=None,X=None):
        raise NotImplementedError