from Semi_sklearn.Algorithm.Classifier.LapSVM import LapSVM
from Semi_sklearn.Evaluation.Classification.Recall import Recall
from Semi_sklearn.Evaluation.Classification.Accuracy import Accuracy
from Semi_sklearn.Dataset.Table.BreastCancer import BreastCancer
f = open("../Result/LapSVM.txt", "w")
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
model=LapSVM(neighbor_mode='connectivity',
           gamma_d=0.03,
           n_neighbor= 5,
           gamma_k=0.03,
           gamma_A= 0.03,
           gamma_I= 0)
model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
result=model.predict(test_X)
print('Accuracy',file=f)
print(Accuracy().scoring(test_y,result),file=f)
print('Recall',file=f)
print(Recall().scoring(test_y,result),file=f)