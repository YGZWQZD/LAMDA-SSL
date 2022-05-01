from Semi_sklearn.Evaluation.Cluster.EvaluationCluster import EvaluationCluster
from sklearn.metrics import davies_bouldin_score
from Semi_sklearn.utils import partial
class Davies_Bouldin_Score(EvaluationCluster):
    def __init__(self):
        super().__init__()
        self.score=davies_bouldin_score
    def scoring(self,y_true=None,clusters=None,X=None):
        # print(clusters)
        return self.score(labels=clusters,X=X)