from Semi_sklearn.Algorithm.Classifier.Assemble import Assemble
from Semi_sklearn.Evaluation.Classification.Accuracy import Accuracy
from Semi_sklearn.Dataset.Table.BreastCancer import BreastCancer
from Semi_sklearn.Evaluation.Classification.Recall import Recall
f = open("../Result/Assemble.txt", "w")
dataset=BreastCancer(test_size=0.3,labeled_size=0.1,stratified=True,shuffle=True,random_state=0)

labeled_X=dataset.pre_transform.fit_transform(dataset.labeled_X)
labeled_y=dataset.labeled_y
unlabeled_X=dataset.pre_transform.fit_transform(dataset.unlabeled_X)
unlabeled_y=dataset.unlabeled_y
test_X=dataset.pre_transform.fit_transform(dataset.test_X)
test_y=dataset.test_y

# print(breast_cancer.target)
from sklearn.svm import SVC
SVM=SVC(C=1.0,kernel='linear',probability=True,gamma='auto')
model=Assemble(T=100000)

model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
result=model.predict(X=test_X)
print('Result',file=f)
print(result,file=f)
print('Accuracy',file=f)
print(Accuracy().scoring(test_y,result),file=f)
print('Recall',file=f)
print(Recall().scoring(test_y,result),file=f)
