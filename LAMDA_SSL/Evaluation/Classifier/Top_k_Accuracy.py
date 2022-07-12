from LAMDA_SSL.utils import partial
from LAMDA_SSL.Base.ClassifierEvaluation import ClassifierEvaluation
from sklearn.metrics import top_k_accuracy_score
from LAMDA_SSL.utils import class_status

class Top_k_Accurary(ClassifierEvaluation):
    def __init__(self,k=2, normalize=True, sample_weight=None, labels=None):
        # >> Parameter
        # >> - k: The k value of Top_k_accurary.
        # >> - normalize: If False, returns the number of correctly classified samples.
        # >> - sample_weight: The weight of each sample.
        # >> - labels: The set of contained labels.
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

