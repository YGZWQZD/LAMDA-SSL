from LAMDA_SSL.Base.RegressorEvaluation import RegressorEvaluation
from sklearn.metrics import mean_absolute_error
from LAMDA_SSL.utils import partial

class Mean_Absolute_Error(RegressorEvaluation):
    def __init__(self,sample_weight=None, multioutput="uniform_average"):
        # >> Parameter
        # >> - sample_weight: The weight of each sample.
        # >> - multioutput: Aggregation method for multiple outputs.
        super().__init__()
        self.sample_weight=sample_weight
        self.multioutput=multioutput
        self.score=partial(mean_absolute_error,sample_weight=self.sample_weight,multioutput=self.multioutput)
    def scoring(self,y_true,y_pred=None):
        return self.score(y_true=y_true,y_pred=y_pred)