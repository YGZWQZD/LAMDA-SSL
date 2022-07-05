from LAMDA_SSL.Evaluation.Regression.EvaluationRegressor import EvaluationRegressor
from sklearn.metrics import mean_absolute_error
from LAMDA_SSL.utils import partial
class Mean_Absolute_Error(EvaluationRegressor):
    def __init__(self,sample_weight=None, multioutput="uniform_average"):
        super().__init__()
        self.sample_weight=sample_weight
        self.multioutput=multioutput
        self.score=partial(mean_absolute_error,sample_weight=self.sample_weight,multioutput=self.multioutput)
    def scoring(self,y_true,y_pred=None):
        return self.score(y_true=y_true,y_pred=y_pred)