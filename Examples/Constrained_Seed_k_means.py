from Semi_sklearn.Algorithm.Cluster.Constrained_Seed_k_means import Constrained_Seed_k_means
import numpy as np

from Semi_sklearn.Evaluation.Cluster.Davies_Bouldin_Score import Davies_Bouldin_Score
from Semi_sklearn.Evaluation.Cluster.Fowlkes_Mallows_Score import Fowlkes_Mallows_Score
from Semi_sklearn.Dataset.Table.Wine import Wine
f = open("../Result/Constrained seed k means.txt", "w")
dataset = Wine(labeled_size=0.2, stratified=True, shuffle=True,random_state=0)
dataset.init_dataset()
dataset.init_transforms()
model = Constrained_Seed_k_means(k=3)
labeled_X = dataset.pre_transform.fit_transform(dataset.labeled_X)
labeled_y = dataset.labeled_y
unlabeled_X = dataset.pre_transform.fit_transform(dataset.unlabeled_X)
unlabeled_y = dataset.unlabeled_y
FLAG=False
while FLAG is not True:
    try:
        model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
        FLAG=True
    except:
        FLAG=False
result=model.predict(unlabeled_X,Transductive=True)
db=Davies_Bouldin_Score().scoring(clusters=result,X=np.vstack((labeled_X,unlabeled_X)))
fm=Fowlkes_Mallows_Score().scoring(clusters=result,y_true=np.hstack((labeled_y,unlabeled_y)))
print("DB Index",file=f)
print(db,file=f)
print("FM Index",file=f)
print(fm,file=f)
