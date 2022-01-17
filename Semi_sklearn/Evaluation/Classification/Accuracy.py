from Semi_sklearn.Evaluation.Classification.EvaluationClassification import EvaluationClassification
from sklearn.metrics import accuracy_score
from Semi_sklearn.utils import partial
class Accuracy(EvaluationClassification):
    def __init__(self,normalize=True, sample_weight=None):
        super().__init__()
        self.normalize=normalize
        self.sample_weight=sample_weight
        self.score=partial(accuracy_score,normalize=self.normalize,sample_weight=self.sample_weight)
    def scoring(self,y_true,y_pred=None,y_score=None):
        return self.score(y_true=y_true,y_pred=y_pred)