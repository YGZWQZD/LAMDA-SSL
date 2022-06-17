from lamda_ssl.Algorithm.Cluster.Constrained_k_means import Constrained_k_means
import numpy as np
from sklearn import datasets
from lamda_ssl.Evaluation.Cluster.Davies_Bouldin_Score import Davies_Bouldin_Score
from lamda_ssl.Evaluation.Cluster.Fowlkes_Mallows_Score import Fowlkes_Mallows_Score
from lamda_ssl.Dataset.Table.Wine import Wine
f = open("../Result/Constrained k means.txt", "w")
dataset=Wine(labeled_size=0.2,stratified=True,shuffle=True,random_state=0)
# breast_cancer = datasets.load_boston()
# rng = np.random.RandomState(55)

# random_unlabeled_points = rng.rand(breast_cancer.target.shape[0]) < 0.3

labeled_X=dataset.pre_transform.fit_transform(dataset.labeled_X)
labeled_y=dataset.labeled_y
unlabeled_X=dataset.pre_transform.fit_transform(dataset.unlabeled_X)
unlabeled_y=dataset.unlabeled_y
# test_X=dataset.pre_transform.fit_transform(dataset.test_X)
# test_y=dataset.test_y
# clusters={}

# for _ in range(3):
#     clusters[_]=set()
# for _ in range(len(labeled_X)):
#     clusters[labeled_y[_]].add(_)
# model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
model=Constrained_k_means(k=3)
ml=[]
cl=[]
for i in range(labeled_X.shape[0]):
    for j in range(i+1,labeled_X.shape[0]):
        if labeled_y[i]==labeled_y[j]:
            ml.append({i,j})
        else:
            cl.append({i,j})
model.fit(X=np.vstack((labeled_X,unlabeled_X)),ml=ml,cl=cl)
result=model.predict(unlabeled_X,Transductive=True)
# print(result)
print("DB Index",file=f)
print(Davies_Bouldin_Score().scoring(clusters=result,X=np.vstack((labeled_X,unlabeled_X))),file=f)
print("FM Index",file=f)
print(Fowlkes_Mallows_Score().scoring(clusters=result,y_true=np.hstack((labeled_y,unlabeled_y))),file=f)