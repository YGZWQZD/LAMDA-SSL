from sklearn.svm import SVC
random_state=None
base_estimator = SVC(C=1.0,kernel='rbf',probability=True,gamma='auto')
base_estimator_2 = SVC(C=1.0,kernel='rbf',probability=True,gamma='auto')
p = 25
n = 25
k = 50
s = 80
binary=False
threshold=0.5
evaluation = None
verbose = False
file = None