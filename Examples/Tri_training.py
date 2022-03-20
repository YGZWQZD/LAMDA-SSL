from Semi_sklearn.Model.Classifier.Tri_training import TriTraining
from sklearn.metrics import accuracy_score
from sklearn import datasets
import numpy as np
from sklearn.svm import SVC
breast_cancer = datasets.load_breast_cancer()
rng = np.random.RandomState(55)

random_unlabeled_points = rng.rand(breast_cancer.target.shape[0]) < 0.3

labeled_X=breast_cancer.data[random_unlabeled_points]
labeled_y=breast_cancer.target[random_unlabeled_points]
unlabeled_X=breast_cancer.data[~random_unlabeled_points]
model=TriTraining(base_estimator=SVC(C=1.0,kernel='linear',probability=True,gamma='auto'))
model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
result=model.predict(unlabeled_X)
print(result)
unlabeled_y=breast_cancer.target[~random_unlabeled_points]
print(unlabeled_y)
print(accuracy_score(unlabeled_y,result))