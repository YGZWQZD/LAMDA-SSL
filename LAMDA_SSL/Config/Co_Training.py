from sklearn.svm import SVC

base_estimator = SVC(C=1.0,kernel='linear',probability=True,gamma='auto')
base_estimator_2 = SVC(C=1.0,kernel='linear',probability=True,gamma='auto')
p = 5
n = 5
k = 30
s = 75
evaluation = None
verbose = False
file = None