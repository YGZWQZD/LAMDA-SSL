from LAMDA_SSL.Base.ClassifierEvaluation import ClassifierEvaluation
from sklearn.metrics import accuracy_score
from LAMDA_SSL.utils import partial

class Accuracy(ClassifierEvaluation):
    def __init__(self,normalize=True, sample_weight=None):
        # >> Parameter
        # >> - normalize: If False, returns the number of correctly classified samples.
        # >> - sample_weight: The weight of each sample.
        super().__init__()
        self.normalize=normalize
        self.sample_weight=sample_weight
        self.score=partial(accuracy_score,normalize=self.normalize,sample_weight=self.sample_weight)
    def scoring(self,y_true,y_pred=None,y_score=None):
        # >> scoring(y_true,y_pred=None,y_score=None): Initialize the data transformation method.
        # >> - y_true: Ground-truth labels.
        # >> - y_pred: Hard labels for model predictions.
        # >> - y_score: Soft labels for model predictions.
        return self.score(y_true=y_true,y_pred=y_pred)