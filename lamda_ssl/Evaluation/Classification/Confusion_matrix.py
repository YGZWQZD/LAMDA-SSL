from lamda_ssl.utils import partial
from lamda_ssl.Evaluation.Classification.EvaluationClassification import EvaluationClassification
from sklearn.metrics import confusion_matrix
class Confusion_matrix(EvaluationClassification):
    def __init__(self,labels=None, sample_weight=None, normalize=None):
        super().__init__()

        self.labels=labels
        self.sample_weight = sample_weight
        self.normalize = normalize
        self.score=partial(confusion_matrix,labels=self.labels,
                           sample_weight=self.sample_weight,
                           normalize=self.normalize)
    def scoring(self,y_true,y_pred=None,y_score=None):
        return self.score(y_true=y_true,y_pred=y_pred)

