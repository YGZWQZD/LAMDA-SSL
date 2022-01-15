from Semi_sklearn.Evaluation.Classification.EvaluationClassification import EvaluationClassification
from sklearn.metrics import roc_auc_score
from Semi_sklearn.utils import partial
class AUC(EvaluationClassification):
    def __init__(self,
                 average="macro",
                 sample_weight=None,
                 max_fpr=None,
                 multi_class="raise",
                 labels=None,):
        super().__init__()
        self.average=average
        self.sample_weight=sample_weight
        self.max_fpr=max_fpr
        self.multi_class=multi_class
        self.lables=labels
        self.score=partial(roc_auc_score,average=self.average,
                           sample_weight=self.sample_weight,max_fpr=self.max_fpr,
                           multi_class=self.multi_class,labels=self.lables)
    def scoring(self,y_true,y_pred=None,y_score=None):
        return self.score(y_true=y_true,y_score=y_score)