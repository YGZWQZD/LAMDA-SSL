from lamda_ssl.utils import partial
from lamda_ssl.Evaluation.Classification.EvaluationClassification import EvaluationClassification
from sklearn.metrics import top_k_accuracy_score
from lamda_ssl.utils import class_status
class Top_k_Accurary(EvaluationClassification):
    def __init__(self,k=2, normalize=True, sample_weight=None, labels=None):
        super().__init__()
        self.k=k
        self.normalize=normalize
        self.sample_weight=sample_weight
        self.labels=labels
        self.score=partial(top_k_accuracy_score,k=self.k,normalize=self.normalize,
                           sample_weight=self.sample_weight,labels=self.labels)
    def scoring(self,y_true,y_pred=None,y_score=None):
        num_classes=class_status(y=y_true).num_classes
        if num_classes==2 and len(y_score.shape)==2:
            return self.score(y_true=y_true, y_score=y_score[:,1])
        else:
            return self.score(y_true=y_true,y_score=y_score)

