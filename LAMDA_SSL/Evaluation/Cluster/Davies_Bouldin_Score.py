from LAMDA_SSL.Base.ClusterEvaluation import ClusterEvaluation
from sklearn.metrics import davies_bouldin_score

class Davies_Bouldin_Score(ClusterEvaluation):
    def __init__(self):
        super().__init__()
        self.score=davies_bouldin_score

    def scoring(self,y_true=None,clusters=None,X=None):
        return self.score(labels=clusters,X=X)