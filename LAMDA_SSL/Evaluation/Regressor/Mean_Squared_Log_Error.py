from LAMDA_SSL.Base.RegressorEvaluation import RegressorEvaluation
from sklearn.metrics import mean_squared_log_error
from LAMDA_SSL.utils import partial

class Mean_Squared_Log_Error(RegressorEvaluation):
    def __init__(self,sample_weight=None, multioutput="uniform_average",squared=True):
        # >> Parameter
        # >> - sample_weight: The weight of each sample.
        # >> - multioutput: Aggregation method for multiple outputs.
        # >> - squared: If True, output the MSLE loss, otherwise output the RMSLE loss.
        super().__init__()
        self.sample_weight=sample_weight
        self.multioutput=multioutput
        self.score=partial(mean_squared_log_error,sample_weight=self.sample_weight,
                           multioutput=self.multioutput,squared=squared)
    def scoring(self,y_true,y_pred=None):
        return self.score(y_true=y_true,y_pred=y_pred)