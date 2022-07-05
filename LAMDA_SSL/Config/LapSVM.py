from LAMDA_SSL.Evaluation.Classification.Recall import Recall
from LAMDA_SSL.Evaluation.Classification.Accuracy import Accuracy
from LAMDA_SSL.Evaluation.Classification.F1 import F1
from LAMDA_SSL.Evaluation.Classification.Precision import Precision
from LAMDA_SSL.Evaluation.Classification.AUC import AUC
from LAMDA_SSL.Evaluation.Classification.Confusion_Matrix import Confusion_Matrix
from sklearn.metrics.pairwise import rbf_kernel

neighbor_mode='connectivity'
distance_function= rbf_kernel,
gamma_d=0.03
t=1
n_neighbor= 5
kernel_function= rbf_kernel
gamma_k=0.03
gamma_A= 0.03
gamma_I= 0

evaluation={
    'accuracy':Accuracy(),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_Matrix(normalize='true')
}

verbose=False
file=None