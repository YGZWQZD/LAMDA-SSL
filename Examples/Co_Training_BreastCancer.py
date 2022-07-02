from lamda_ssl.Algorithm.Classifier.Co_Training import Co_Training
from lamda_ssl.Dataset.Table.BreastCancer import BreastCancer
from lamda_ssl.Evaluation.Classification.Accuracy import Accuracy
from lamda_ssl.Evaluation.Classification.Precision import Precision
from lamda_ssl.Evaluation.Classification.Recall import Recall
from lamda_ssl.Evaluation.Classification.F1 import F1
from lamda_ssl.Evaluation.Classification.AUC import AUC
from lamda_ssl.Evaluation.Classification.Confusion_Matrix import Confusion_Matrix
from lamda_ssl.Split.View_Split import View_Split
import numpy as np
from sklearn.svm import SVC

file = open("../Result/Co_Training_BreastCancer.txt", "w")

dataset=BreastCancer(test_size=0.3,labeled_size=0.1,stratified=True,shuffle=True,random_state=0,default_transforms=True)

labeled_X=dataset.labeled_X
labeled_y=dataset.labeled_y
unlabeled_X=dataset.unlabeled_X
unlabeled_y=dataset.unlabeled_y
test_X=dataset.test_X
test_y=dataset.test_y

pre_transform=dataset.pre_transform
pre_transform.fit(np.vstack([labeled_X, unlabeled_X]))

labeled_X=pre_transform.transform(labeled_X)
unlabeled_X=pre_transform.transform(unlabeled_X)
test_X=pre_transform.transform(test_X)

# View split
split_labeled_X=View_Split(labeled_X,mode='sequential')
split_unlabeled_X=View_Split(unlabeled_X,mode='sequential')
split_test_X=View_Split(test_X,mode='sequential')

# Base estimator
SVM=SVC(C=1.0,kernel='linear',probability=True,gamma='auto')

evaluation={
    'accuracy':Accuracy(),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_Matrix(normalize='true')
}

model=Co_Training(base_estimator=SVM,s=(len(labeled_X)+len(unlabeled_X))//10,evaluation=evaluation,verbose=True,file=file)

model.fit(X=split_labeled_X,y=labeled_y,unlabeled_X=split_unlabeled_X)

performance=model.evaluate(X=split_test_X,y=test_y)

result=model.y_pred

print(result,file=file)

print(performance,file=file)