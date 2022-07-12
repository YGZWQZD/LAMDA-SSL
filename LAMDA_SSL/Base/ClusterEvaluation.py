from abc import ABC,abstractmethod
class ClusterEvaluation(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def scoring(self,y_true=None,clusters=None,X=None):
        # > scoring(y_true=None,clusters=None,X=None): Initialize the data transformation method.
        # >> - y_true: Ground-truth labels.
        # >> - clusters: Clustering results.
        # >> - X: Sample features used in clustering.
        raise NotImplementedError