from Semi_sklearn.Model.Classifier.TSVM import TSVM
import numpy as np
from sklearn import datasets

breast_cancer = datasets.load_breast_cancer()
rng = np.random.RandomState(55)

random_unlabeled_points = rng.rand(breast_cancer.target.shape[0]) < 0.3

labeled_X=breast_cancer.data[random_unlabeled_points]
labeled_y=breast_cancer.target[random_unlabeled_points]
unlabeled_X=breast_cancer.data[~random_unlabeled_points]
# print(breast_cancer.target)
model=TSVM(Cl=15,Cu=0.0001,kernel='linear')


model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
unlabeled_y=model.predict(X=unlabeled_X,Transductive=False)
print(unlabeled_y)
