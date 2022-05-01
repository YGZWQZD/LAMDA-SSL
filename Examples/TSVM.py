from Semi_sklearn.Alogrithm.Classifier.TSVM import TSVM
from Semi_sklearn.Evaluation.Classification.Recall import Recall
from Semi_sklearn.Evaluation.Classification.Accuracy import Accuracy
from Semi_sklearn.Dataset.Table.BreastCancer import BreastCancer
f = open("../Result/TSVM.txt", "w")
dataset=BreastCancer(test_size=0.3,labeled_size=0.1,stratified=True,shuffle=True,random_state=0)
dataset.init_dataset()
dataset.init_transforms()

labeled_X=dataset.pre_transform.fit_transform(dataset.labeled_X)
labeled_y=dataset.labeled_y
unlabeled_X=dataset.pre_transform.fit_transform(dataset.unlabeled_X)
unlabeled_y=dataset.unlabeled_y
test_X=dataset.pre_transform.fit_transform(dataset.test_X)
test_y=dataset.test_y
model=TSVM(Cl=15,Cu=0.0001,kernel='linear')


model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
result=model.predict(test_X,Transductive=False)
# print(result)
# print(test_y)
print('Accuracy',file=f)
print(Accuracy().scoring(test_y,result),file=f)
print('Recall',file=f)
print(Recall().scoring(test_y,result),file=f)

