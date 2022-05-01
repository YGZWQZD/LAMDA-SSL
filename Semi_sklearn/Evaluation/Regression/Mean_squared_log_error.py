from Semi_sklearn.Evaluation.Regression.EvaluationRegressor import EvaluationRegressor
from sklearn.metrics import mean_squared_log_error
from Semi_sklearn.utils import partial
class Mean_squared_log_error(EvaluationRegressor):
    def __init__(self,sample_weight=None, multioutput="uniform_average",squared=True):
        super().__init__()
        self.sample_weight=sample_weight
        self.multioutput=multioutput
        self.score=partial(mean_squared_log_error,sample_weight=self.sample_weight,
                           multioutput=self.multioutput,squared=squared)
    def scoring(self,y_true,y_pred=None):
        return self.score(y_true=y_true,y_pred=y_pred)