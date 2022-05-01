from Semi_sklearn.Alogrithm.Classifier.SSGMM import SSGMM
import numpy as np
from sklearn import datasets
from Semi_sklearn.Evaluation.Classification.Accuracy import Accuracy
from Semi_sklearn.Evaluation.Classification.Recall import Recall
from Semi_sklearn.Dataset.Table.BreastCancer import BreastCancer
f = open("../Result/SSGMM.txt", "w")
dataset=BreastCancer(test_size=0.3,labeled_size=0.1,stratified=True,shuffle=True,random_state=0)
dataset.init_dataset()
dataset.init_transforms()
# breast_cancer = datasets.load_boston()
# rng = np.random.RandomState(55)

# random_unlabeled_points = rng.rand(breast_cancer.target.shape[0]) < 0.3

labeled_X=dataset.pre_transform.fit_transform(dataset.labeled_X)
labeled_y=dataset.labeled_y
unlabeled_X=dataset.pre_transform.fit_transform(dataset.unlabeled_X)
unlabeled_y=dataset.unlabeled_y
test_X=dataset.pre_transform.fit_transform(dataset.test_X)
test_y=dataset.test_y
model=SSGMM(n_class=2,tolerance=0.000001)
model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
result=model.predict(test_X)
# print(result)
# print(test_y)
print('Accuracy',file=f)
print(Accuracy().scoring(test_y,result),file=f)
print('Recall',file=f)
print(Recall().scoring(test_y,result),file=f)