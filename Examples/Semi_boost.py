from Semi_sklearn.Model.Classifier.SemiBoost import SemiBoostClassifier
import numpy as np
from sklearn import datasets
from Semi_sklearn.Evaluation.Classification.Accuracy import Accuracy
from sklearn.preprocessing import StandardScaler

breast_cancer = datasets.load_breast_cancer()
rng = np.random.RandomState(55)

random_unlabeled_points = rng.rand(breast_cancer.target.shape[0]) < 0.3

labeled_X=breast_cancer.data[random_unlabeled_points]
labeled_y=breast_cancer.target[random_unlabeled_points]
unlabeled_X=breast_cancer.data[~random_unlabeled_points]
# print(labeled_y)
# print(breast_cancer.target)
from sklearn.svm import SVC
SVM=SVC(C=1.0,kernel='linear',probability=True,gamma='auto')
model=SemiBoostClassifier()
labeled_X=StandardScaler().fit_transform(labeled_X)
unlabeled_X=StandardScaler().fit_transform(unlabeled_X)
# print(labeled_X.shape)
# print(unlabeled_X.shape)

model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
result=model.predict(X=unlabeled_X)
# print(unlabeled_y)
unlabeled_y=breast_cancer.target[~random_unlabeled_points]
print(Accuracy().scoring(unlabeled_y,result))
