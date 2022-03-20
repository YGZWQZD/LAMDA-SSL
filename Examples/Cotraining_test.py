from Semi_sklearn.Model.Classifier.Co_training import Co_training
import numpy as np
from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn.metrics import  accuracy_score

breast_cancer = datasets.load_breast_cancer()
rng = np.random.RandomState(55)

random_unlabeled_points = rng.rand(breast_cancer.target.shape[0]) < 0.3

labeled_X=breast_cancer.data[random_unlabeled_points]
labeled_y=breast_cancer.target[random_unlabeled_points]
unlabeled_X=breast_cancer.data[~random_unlabeled_points]

X,X_2=labeled_X[:,:labeled_X.shape[1]//2],labeled_X[:,labeled_X.shape[1]//2:]

unlabeled_X,unlabeled_X_2=unlabeled_X[:,:unlabeled_X.shape[1]//2],unlabeled_X[:,unlabeled_X.shape[1]//2:]

# print(breast_cancer.target)
from sklearn.svm import SVC
SVM=SVC(C=1.0,kernel='linear',probability=True,gamma='auto')
# model=Co_training(base_estimator=SVM,threshold=0.75,criterion="threshold",max_iter=100)
#
#
# unlabeled_y=model.predict(X=unlabeled_X)
# print(unlabeled_y)

model=Co_training(base_estimator=SVM,s=(len(X)+len(unlabeled_X))//10)
model.fit(X=(X,X_2),y=labeled_y,unlabeled_X=(unlabeled_X,unlabeled_X_2))
result=model.predict(X=(unlabeled_X,unlabeled_X_2))
unlabeled_y=breast_cancer.target[~random_unlabeled_points]
print(accuracy_score(unlabeled_y,result))