from sklearn.metrics import silhouette_score
from LAMDA_SSL.Evaluation.Cluster.EvaluationCluster import EvaluationCluster
from LAMDA_SSL.utils import partial

class Silhouette_Score(EvaluationCluster):
    def __init__(self, metric="euclidean", sample_size=None, random_state=None):
        super().__init__()
        self.score=partial(silhouette_score, metric=metric, sample_size=sample_size, random_state=random_state)
    def scoring(self,y_true=None,clusters=None,X=None):
        return self.score(labels=clusters,X=X)