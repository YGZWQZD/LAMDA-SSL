from LAMDA_SSL.Evaluation.Classification.Accuracy import Accuracy
from LAMDA_SSL.Evaluation.Classification.Recall import Recall
from LAMDA_SSL.Evaluation.Classification.F1 import F1
from LAMDA_SSL.Evaluation.Classification.Precision import Precision
from LAMDA_SSL.Evaluation.Classification.AUC import AUC
from LAMDA_SSL.Evaluation.Classification.Confusion_Matrix import Confusion_Matrix
from sklearn.svm import SVC

base_estimator = SVC(C=1.0,kernel='linear',probability=True,gamma='auto')
n_neighbors=4
n_jobs = 1
max_models = 300
sample_percent = 0.01
sigma_percentile = 90
similarity_kernel = 'rbf'
gamma=0.001
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