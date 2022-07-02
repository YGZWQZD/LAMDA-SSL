from lamda_ssl.Evaluation.Classification.Accuracy import Accuracy
from lamda_ssl.Evaluation.Classification.Precision import Precision
from lamda_ssl.Evaluation.Classification.Recall import Recall
from lamda_ssl.Evaluation.Classification.F1 import F1
from lamda_ssl.Evaluation.Classification.AUC import AUC
from lamda_ssl.Evaluation.Classification.Confusion_Matrix import Confusion_Matrix

Cl = 15
Cu = 0.0001
kernel = 'linear'
degree = 3
gamma = "scale"
coef0 = 0.0
shrinking = True
probability = True
tol = 1e-3
cache_size = 200
class_weight = None
max_iter = -1
decision_function_shape = "ovr"
break_ties = False
random_state = None
evaluation={
    'accuracy':Accuracy(),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_Matrix(normalize='true')
}
verbose = False
file = None