from sklearn.linear_model import LogisticRegression
from lamda_ssl.Evaluation.Classification.Precision import Precision
from lamda_ssl.Evaluation.Classification.Recall import Recall
from lamda_ssl.Evaluation.Classification.F1 import F1
from lamda_ssl.Evaluation.Classification.AUC import AUC
from lamda_ssl.Evaluation.Classification.Top_k_Accuracy import Top_k_Accurary
from lamda_ssl.Evaluation.Classification.Confusion_Matrix import Confusion_Matrix
from lamda_ssl.Evaluation.Classification.Accuracy import Accuracy
from lamda_ssl.Scheduler.StepLR import StepLR
from lamda_ssl.Opitimizer.Adam import Adam

epoch = 1000
eval_epoch = 10
optimizer = Adam(lr=0.001)
scheduler = StepLR(step_size=10,gamma=0.9,verbose=False)
device = 'cpu'
evaluation={
    'accuracy':Accuracy(),
    'top_5_accuracy':Top_k_Accurary(k=5),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_Matrix(normalize='true')
}
weight_decay = 0
network = None
parallel = None
file = None
verbose = False
xeqs = True
num_features = None
num_nodes = None
hidden_layers = [1000,1000]
alpha = 1e-2
beta = 5
gamma = 1e-5
base_estimator = LogisticRegression()