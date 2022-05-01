from Semi_sklearn.Alogrithm.Classifier.Co_training import Co_training
from Semi_sklearn.Evaluation.Classification.Accuracy import Accuracy
from Semi_sklearn.Dataset.Table.BreastCancer import BreastCancer
from Semi_sklearn.Evaluation.Classification.Recall import Recall
f = open("../Result/Co-Training.txt", "w")
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

labeled_X_1,labeled_X_2=labeled_X[:,:labeled_X.shape[1]//2],labeled_X[:,labeled_X.shape[1]//2:]

unlabeled_X_1,unlabeled_X_2=unlabeled_X[:,:unlabeled_X.shape[1]//2],unlabeled_X[:,unlabeled_X.shape[1]//2:]

test_X_1,test_X_2=test_X[:,:test_X.shape[1]//2],test_X[:,test_X.shape[1]//2:]
# print(breast_cancer.target)
from sklearn.svm import SVC
SVM=SVC(C=1.0,kernel='linear',probability=True,gamma='auto')
# model=Co_training(base_estimator=SVM,threshold=0.75,criterion="threshold",max_iter=100)
#
#
# unlabeled_y=model.predict(X=unlabeled_X)
# print(unlabeled_y)

model=Co_training(base_estimator=SVM,s=(len(labeled_X)+len(unlabeled_X))//10)
model.fit(X=(labeled_X_1,labeled_X_2),y=labeled_y,unlabeled_X=(unlabeled_X_1,unlabeled_X_2))
result=model.predict(X=(test_X_1,test_X_2))
print('Accuracy',file=f)
print(Accuracy().scoring(test_y,result),file=f)
print('Recall',file=f)
print(Recall().scoring(test_y,result),file=f)