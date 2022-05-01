from Semi_sklearn.Alogrithm.Classifier.SemiGMM import SemiGMM
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
breast_cancer = datasets.load_digits()

rng = np.random.RandomState(54)

random_unlabeled_points = rng.rand(breast_cancer.target.shape[0]) < 0.3

from sklearn import preprocessing
data=preprocessing.MinMaxScaler().fit_transform(breast_cancer.data)

labeled_X=data[random_unlabeled_points]
labeled_y=breast_cancer.target[random_unlabeled_points]

unlabeled_X=data[~random_unlabeled_points]
model=SemiGMM(n_class=10,tolerance=0.000001)
model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
result=model.predict(unlabeled_X)
unlabeled_y=breast_cancer.target[~random_unlabeled_points]
print(accuracy_score(unlabeled_y,result))