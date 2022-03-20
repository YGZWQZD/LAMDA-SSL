from Semi_sklearn.Model.Classifier.Self_training import Self_training
import numpy as np
from sklearn import datasets

breast_cancer = datasets.load_breast_cancer()
rng = np.random.RandomState(55)

random_unlabeled_points = rng.rand(breast_cancer.target.shape[0]) < 0.3

labeled_X=breast_cancer.data[random_unlabeled_points]
labeled_y=breast_cancer.target[random_unlabeled_points]
unlabeled_X=breast_cancer.data[~random_unlabeled_points]

# print(breast_cancer.target)
from sklearn.svm import SVC
SVM=SVC(C=1.0,kernel='linear',probability=True,gamma='auto')
model=Self_training(base_estimator=SVM,threshold=0.75,criterion="threshold",max_iter=100)

model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
unlabeled_y=model.predict(X=unlabeled_X)
print(unlabeled_y)
