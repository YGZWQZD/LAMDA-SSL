from sklearn.metrics import silhouette_score
from LAMDA_SSL.Base.ClusterEvaluation import ClusterEvaluation
from LAMDA_SSL.utils import partial

class Silhouette_Score(ClusterEvaluation):
    def __init__(self, metric="euclidean", sample_size=None, random_state=None):
        # >> Parameter
        # >> - metric : The metric to use when calculating distance between instances in a feature array. If metric is a string, it must be one of the options allowed by <sklearn.metrics.pairwise.pairwise_distances>. If ``X`` of the `scoring` method is the distance array itself, use ``metric="precomputed"``.
        # >> - sample_size: The size of the sample to use when computing the Silhouette Coefficient on a random subset of the data.
        # >> - random_state : Determines random number generation for selecting a subset of samples.
        super().__init__()
        self.score=partial(silhouette_score, metric=metric, sample_size=sample_size, random_state=random_state)
    def scoring(self,y_true=None,clusters=None,X=None):
        return self.score(labels=clusters,X=X)