from Semi_sklearn.Dataset.Graph.Cora import Cora
from Semi_sklearn.Evaluation.Classification.Precision import Precision
from Semi_sklearn.Evaluation.Classification.Recall import Recall
from Semi_sklearn.Evaluation.Classification.F1 import F1
from Semi_sklearn.Evaluation.Classification.Accuracy import Accuracy

from Semi_sklearn.Scheduler.StepLR import StepLR
from Semi_sklearn.Opitimizer.Adam import Adam
optimizer=Adam(lr=0.001)
LR=StepLR(step_size=10,gamma=0.9)
f = open("../Result/SDNE.txt", "w")
dataset=Cora(labeled_size=0.2,root='..\Semi_sklearn\Download\Cora',random_state=0)

data=dataset.data
from Semi_sklearn.Algorithm.Classifier.SDNE import SDNE
model=SDNE(
    num_nodes=data.x.shape[0],
    input_dim=data.x.shape[1],
    hidden_layers=[1000,1000],
    epoch=300,
    gamma=1e-5,
    optimizer=optimizer,
    scheduler=LR,
    device='cpu',
    weight_decay=0,
    file=f
)
model.fit(data,valid_X=data.unlabeled_mask)
# adj=model.adjacency_matrix
# print(model.embedding.shape)
#
# y_pred=model.predict(X=data.unlabeled_mask)
# y=data.y[data.unlabeled_mask]
# print(Accuracy().scoring(y,y_pred=y_pred))
# from sklearn.neighbors import KNeighborsClassifier
# model=KNeighborsClassifier()
# model.fit(adj[data.labeled_mask].numpy(),y=data.y[data.labeled_mask].numpy())
y_pred=model.predict(data.unlabeled_mask)
y=data.y[data.unlabeled_mask]
model.fit(data,valid_X=data.unlabeled_mask)
# adj=model.adjacency_matrix
print('Accuracy',file=f)
print(Accuracy().scoring(y,y_pred=y_pred),file=f)
print('Recall',file=f)
print(Recall(average='macro').scoring(y,y_pred=y_pred),file=f)
print('Precision',file=f)
print(Precision(average='macro').scoring(y,y_pred=y_pred),file=f)