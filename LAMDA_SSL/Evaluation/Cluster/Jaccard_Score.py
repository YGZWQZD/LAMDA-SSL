from sklearn.metrics import jaccard_score
from LAMDA_SSL.Base.ClusterEvaluation import ClusterEvaluation
from LAMDA_SSL.utils import partial

class Jaccard_Score(ClusterEvaluation):
    def __init__(self, labels=None, pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn"):
        # >> Parameter
        # >> - labels: The set of labels to include when ``average != 'binary'``, and their order if ``average is None``. Labels present in the data can be excluded, for example to calculate a multiclass average ignoring a majority negative class, while labels not present in the data will result in 0 components in a macro average. For multilabel targets, labels are column indices. By default, all labels in ``y_true`` and ``y_pred`` are used in sorted order.
        #
        # >> - pos_label : The class to report if ``average='binary'`` and the data is binary. If the data are multiclass or multilabel, this will be ignored;
        # setting ``labels=[pos_label]`` and ``average != 'binary'`` will report scores for that label only.
        #
        # >>- average : {'micro', 'macro', 'samples', 'weighted', 'binary'} or None. If ``None``, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data:
        #     ``'binary'``:
        #         Only report results for the class specified by ``pos_label``.
        #         This is applicable only if targets (``y_{true,pred}``) are binary.
        #     ``'micro'``:
        #         Calculate metrics globally by counting the total true positives,
        #         false negatives and false positives.
        #     ``'macro'``:
        #         Calculate metrics for each label, and find their unweighted
        #         mean.  This does not take label imbalance into account.
        #     ``'weighted'``:
        #         Calculate metrics for each label, and find their average, weighted
        #         by support (the number of true instances for each label). This
        #         alters 'macro' to account for label imbalance.
        #     ``'samples'``:
        #         Calculate metrics for each instance, and find their average (only
        #         meaningful for multilabel classification).
        #
        # >> - sample_weight : array-like of shape (n_samples,), Sample weights.
        #
        # >> - zero_division : "warn", {0.0, 1.0}, default="warn" Sets the value to return when there is a zero division, i.e. when there are no negative values in predictions and labels. If set to "warn", this acts like 0, but a warning is also raised.
        super().__init__()
        self.score=partial(jaccard_score,labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division)

    def scoring(self,y_true=None,clusters=None,X=None):
        return self.score(y_true=y_true,y_pred=clusters)