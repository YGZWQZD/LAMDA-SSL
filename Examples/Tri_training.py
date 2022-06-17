from lamda_ssl.Algorithm.Classifier.Tri_training import TriTraining
import numpy as np
from sklearn import datasets
from lamda_ssl.Evaluation.Classification.Accuracy import Accuracy
from lamda_ssl.Dataset.Table.BreastCancer import BreastCancer
from lamda_ssl.Evaluation.Classification.Recall import Recall
f = open("../Result/Tri-Training.txt", "w")
dataset=BreastCancer(test_size=0.3,labeled_size=0.1,stratified=True,shuffle=True,random_state=0)
# breast_cancer = datasets.load_boston()
# rng = np.random.RandomState(55)

# random_unlabeled_points = rng.rand(breast_cancer.target.shape[0]) < 0.3

labeled_X=dataset.pre_transform.fit_transform(dataset.labeled_X)
labeled_y=dataset.labeled_y
unlabeled_X=dataset.pre_transform.fit_transform(dataset.unlabeled_X)
unlabeled_y=dataset.unlabeled_y
test_X=dataset.pre_transform.fit_transform(dataset.test_X)
test_y=dataset.test_y
from sklearn.svm import SVC

model=TriTraining(base_estimator=SVC(C=1.0,kernel='linear',probability=True,gamma='auto'))
model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
result=model.predict(test_X)
print('Accuracy',file=f)
print(Accuracy().scoring(test_y,result),file=f)
print('Recall',file=f)
print(Recall().scoring(test_y,result),file=f)