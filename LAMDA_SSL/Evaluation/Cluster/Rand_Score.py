from sklearn.metrics import rand_score
from LAMDA_SSL.Evaluation.Cluster.EvaluationCluster import EvaluationCluster

class Rand_Score(EvaluationCluster):
    def __init__(self):
        super().__init__()
        self.score=rand_score
    def scoring(self,y_true=None,clusters=None,X=None):
        return self.score(labels_true=y_true,labels_pred=clusters)