from LAMDA_SSL.utils import partial
from LAMDA_SSL.Base.ClassifierEvaluation import ClassifierEvaluation
from sklearn.metrics import confusion_matrix

class Confusion_Matrix(ClassifierEvaluation):
    def __init__(self,labels=None, sample_weight=None, normalize=None):
        # >> Parameter
        # >> - average: The way to calculate the AUC mean, optional 'micro', 'macro', 'samples', 'weighted' or None.
        # >> - sample_weight: The weight of each sample.
        # >> - max_fpr: Used to determine the range when only a partial AUC is calculated.
        # >> - multi_class: Method for handling multiple classes, optional 'raise', 'ovr', 'ovo'.
        # >> - labels: The set of contained labels.
        super().__init__()
        self.labels=labels
        self.sample_weight = sample_weight
        self.normalize = normalize
        self.score=partial(confusion_matrix,labels=self.labels,
                           sample_weight=self.sample_weight,
                           normalize=self.normalize)

    def scoring(self,y_true,y_pred=None,y_score=None):
        return self.score(y_true=y_true,y_pred=y_pred)

