from Semi_sklearn.Alogrithm.Cluster.Constrained_k_means import Constrained_k_means
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import datasets
data=datasets.load_wine()
# model=Constrained_k_means(k=3)

rng = np.random.RandomState(55)
#0,7
#1,2,3
#4,5,6
# model=Constrained_Seed_k_means(k=3)
random_unlabeled_points = rng.rand(data.target.shape[0]) < 0.3

labeled_X=data.data[random_unlabeled_points]
labeled_y=data.target[random_unlabeled_points]
unlabeled_X=data.data[~random_unlabeled_points]
unlabeled_y=data.target[~random_unlabeled_points]
# clusters={}

# for _ in range(3):
#     clusters[_]=set()
# for _ in range(len(labeled_X)):
#     clusters[labeled_y[_]].add(_)
# model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
model=Constrained_k_means(k=3)
model.fit(X=labeled_X,ml=[{1,2},{1,3},{4,5}],cl=[{1,4},{4,7},{0,2},{2,4}])
print(model.predict(unlabeled_X))

print(accuracy_score(unlabeled_y,model.predict(unlabeled_X,Transductive=False)))