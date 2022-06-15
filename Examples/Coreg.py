from Semi_sklearn.Algorithm.Regressor.CoReg import CoReg
from Semi_sklearn.Evaluation.Regression.Mean_absolute_error import Mean_absolute_error
from Semi_sklearn.Evaluation.Regression.Mean_squared_error import MeanSquaredError
from Semi_sklearn.Dataset.Table.Boston import Boston

dataset=Boston(labeled_size=0.3,test_size=0.1,stratified=False,shuffle=True,random_state=0)
f = open("../Result/CoReg.txt", "w")
# breast_cancer = datasets.load_boston()
# rng = np.random.RandomState(55)

# random_unlabeled_points = rng.rand(breast_cancer.target.shape[0]) < 0.3

labeled_X=dataset.pre_transform.fit_transform(dataset.labeled_X)
labeled_y=dataset.labeled_y
unlabeled_X=dataset.pre_transform.fit_transform(dataset.unlabeled_X)
unlabeled_y=dataset.unlabeled_y
test_X=dataset.pre_transform.fit_transform(dataset.test_X)
test_y=dataset.test_y
# print(breast_cancer.target)
# from sklearn.svm import SVC
# SVM=SVC(C=1.0,kernel='linear',probability=True,gamma='auto')
model=CoReg()

model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
result=model.predict(X=test_X)
# unlabeled_y=breast_cancer.target[~random_unlabeled_points]
# print(unlabeled_y)
print(Mean_absolute_error().scoring(test_y,result),file=f)
print(MeanSquaredError().scoring(test_y,result),file=f)
