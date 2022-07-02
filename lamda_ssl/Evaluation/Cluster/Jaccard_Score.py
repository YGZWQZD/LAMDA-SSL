from sklearn.metrics import jaccard_score
from lamda_ssl.Evaluation.Cluster.EvaluationCluster import EvaluationCluster
from lamda_ssl.utils import partial

class Jaccard_Score(EvaluationCluster):
    def __init__(self,    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn"):
        super().__init__()
        self.score=partial(jaccard_score,labels=labels,
    pos_label=pos_label,
    average=average,
    sample_weight=sample_weight,
    zero_division=zero_division)
    def scoring(self,y_true=None,clusters=None,X=None):
        return self.score(y_true=y_true,y_pred=clusters)