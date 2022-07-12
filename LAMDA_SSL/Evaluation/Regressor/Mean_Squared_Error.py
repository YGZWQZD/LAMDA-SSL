from LAMDA_SSL.Base.RegressorEvaluation import RegressorEvaluation
from sklearn.metrics import mean_squared_error
from LAMDA_SSL.utils import partial

class Mean_Squared_Error(RegressorEvaluation):
    def __init__(self,sample_weight=None, multioutput="uniform_average",squared=True):
        # >> Parameter
        # >> - sample_weight: The weight of each sample.
        # >> - multioutput: Aggregation method for multiple outputs.
        # >> - squared: If True, output the MSE loss, otherwise output the RMSE loss.
        super().__init__()
        self.sample_weight=sample_weight
        self.multioutput=multioutput
        self.squared=squared
        self.score=partial(mean_squared_error,sample_weight=self.sample_weight,
                           multioutput=self.multioutput,squared=squared)
    def scoring(self,y_true,y_pred=None):
        return self.score(y_true=y_true,y_pred=y_pred)