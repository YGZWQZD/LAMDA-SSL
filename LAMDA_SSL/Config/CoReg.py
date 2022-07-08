from LAMDA_SSL.Evaluation.Regressor.Mean_Absolute_Error import Mean_Absolute_Error
from LAMDA_SSL.Evaluation.Regressor.Mean_Squared_Error import Mean_Squared_Error
from LAMDA_SSL.Evaluation.Regressor.Mean_Squared_Log_Error import Mean_Squared_Log_Error

k1=3
k2=3
p1=2
p2=5
max_iters=100
pool_size=100
evaluation={
    'Mean_Absolute_Error':Mean_Absolute_Error(),
    'Mean_Squared_Error':Mean_Squared_Error(),
    'Mean_Squared_Log_Error':Mean_Squared_Log_Error()
}
verbose=False
file=None