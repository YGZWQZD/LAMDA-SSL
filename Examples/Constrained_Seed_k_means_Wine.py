from LAMDA_SSL.Algorithm.Clustering.Constrained_Seed_k_means import Constrained_Seed_k_means
from LAMDA_SSL.Evaluation.Cluster.Davies_Bouldin_Score import Davies_Bouldin_Score
from LAMDA_SSL.Evaluation.Cluster.Fowlkes_Mallows_Score import Fowlkes_Mallows_Score
from LAMDA_SSL.Evaluation.Cluster.Jaccard_Score import Jaccard_Score
from LAMDA_SSL.Evaluation.Cluster.Silhouette_Score import Silhouette_Score
from LAMDA_SSL.Evaluation.Cluster.Rand_Score import Rand_Score
from LAMDA_SSL.Dataset.Tabular.Wine import Wine
import numpy as np

file = open("../Result/Constrained_Seed_k_means_Wine.txt", "w")

dataset = Wine(labeled_size=0.2, stratified=True, shuffle=True,random_state=0,default_transforms=True)

labeled_X=dataset.labeled_X
labeled_y=dataset.labeled_y
unlabeled_X=dataset.unlabeled_X
unlabeled_y=dataset.unlabeled_y

pre_transform=dataset.pre_transform
pre_transform.fit(np.vstack([labeled_X, unlabeled_X]))

labeled_X=pre_transform.transform(labeled_X)
unlabeled_X=pre_transform.transform(unlabeled_X)

evaluation={
    'Fowlkes_Mallows_Score':Fowlkes_Mallows_Score(),
    'Jaccard_Score':Jaccard_Score(average='macro'),
    'Rand_Score':Rand_Score(),
    'Davies_Bouldin_Score':Davies_Bouldin_Score(),
    'Silhouette_Score':Silhouette_Score()
}

model = Constrained_Seed_k_means(k=3,evaluation=evaluation,verbose=True,file=file)

model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)

performance=model.evaluate(y=np.hstack([labeled_y, unlabeled_y]),Transductive=True)

result=model.y_pred

print(result,file=file)

print(performance,file=file)
