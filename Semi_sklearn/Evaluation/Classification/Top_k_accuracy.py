from Semi_sklearn.utils import partial
from Semi_sklearn.Evaluation.Classification.EvaluationClassification import EvaluationClassification
from sklearn.metrics import top_k_accuracy_score
class Top_k_accurary(EvaluationClassification):
    def __init__(self,k=2, normalize=True, sample_weight=None, labels=None):
        super().__init__()
        self.k=k
        self.normalize=normalize
        self.sample_weight=sample_weight
        self.labels=labels
        self.score=partial(top_k_accuracy_score,k=self.k,normalize=self.normalize,
                           sample_weight=self.sample_weight,labels=self.labels)
    def scoring(self,y_true,y_pred=None,y_score=None):
        return self.score(y_true=y_true,y_score=y_score)

