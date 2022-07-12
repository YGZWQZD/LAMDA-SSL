from LAMDA_SSL.Base.ClassifierEvaluation import ClassifierEvaluation
from sklearn.metrics import roc_auc_score
from LAMDA_SSL.utils import partial
from LAMDA_SSL.utils import class_status

class AUC(ClassifierEvaluation):
    def __init__(self,
                 average="macro",
                 sample_weight=None,
                 max_fpr=None,
                 multi_class="raise",
                 labels=None):
        # >> Parameter
        # >> - average: The way to calculate the AUC mean, optional 'micro', 'macro', 'samples', 'weighted' or None.
        # >> - sample_weight: The weight of each sample.
        # >> - max_fpr: Used to determine the range when only a partial AUC is calculated.
        # >> - multi_class: Method for handling multiple classes, optional 'raise', 'ovr', 'ovo'.
        # >> - labels: The set of contained labels.
        super().__init__()
        self.average=average
        self.sample_weight=sample_weight
        self.max_fpr=max_fpr
        self.multi_class=multi_class
        self.labels=labels
        self.score=partial(roc_auc_score,average=self.average,
                           sample_weight=self.sample_weight,max_fpr=self.max_fpr,
                           multi_class=self.multi_class,labels=self.labels)


    def scoring(self,y_true,y_pred=None,y_score=None):
        num_classes=class_status(y=y_true).num_classes
        if num_classes==2 and len(y_score.shape)==2:
            return self.score(y_true=y_true, y_score=y_score[:,1])
        else:
            return self.score(y_true=y_true,y_score=y_score)