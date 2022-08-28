from sklearn.svm import SVC
random_state=None
base_estimator = SVC(C=1.0,kernel='rbf',probability=True,gamma='auto')
base_estimator_2 = SVC(C=1.0,kernel='rbf',probability=True,gamma='auto')
p = 30
n = 30
k = 30
s = 100
evaluation = None
verbose = False
file = None