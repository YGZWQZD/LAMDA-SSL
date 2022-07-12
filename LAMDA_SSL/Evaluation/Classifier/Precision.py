from LAMDA_SSL.Base.ClassifierEvaluation import ClassifierEvaluation
from sklearn.metrics import precision_score
from LAMDA_SSL.utils import partial

class Precision(ClassifierEvaluation):
    def __init__(self,labels=None,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn"):
        # >> Parameter
        # >> - labels: The set of contained labels.
        # >> - pos_label: Positive label for binary classification.
        # >> - average: The calculation method for multi-classification, optional 'micro', 'macro', 'samples', 'weighted', 'binary'.
        # >> - sample_weight: The weight of each sample.
        # >> - zero_division: The return value when the denominator is 0.
        super().__init__()
        self.labels=labels
        self.pos_label=pos_label
        self.average=average
        self.sample_weight=sample_weight
        self.zero_division=zero_division
        self.score=partial(precision_score,labels=self.labels,pos_label=self.pos_label,
                           average=self.average,sample_weight=self.sample_weight,
                           zero_division=self.zero_division)
    def scoring(self,y_true,y_pred=None,y_score=None):
        return self.score(y_true=y_true,y_pred=y_pred)