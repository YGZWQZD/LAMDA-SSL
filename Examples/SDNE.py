from Semi_sklearn.Dataset.Graph.Cora import Cora
#
import numpy as np
import torch
from Semi_sklearn.Evaluation.Classification.Accuracy import Accuracy
from Semi_sklearn.Evaluation.Classification.Top_k_accuracy import Top_k_accurary
from Semi_sklearn.Evaluation.Classification.Precision import Precision
from Semi_sklearn.Evaluation.Classification.Recall import Recall
from Semi_sklearn.Evaluation.Classification.F1 import F1
from Semi_sklearn.Evaluation.Classification.Accuracy import Accuracy
from Semi_sklearn.Evaluation.Classification.Confusion_matrix import Confusion_matrix
from Semi_sklearn.Opitimizer.Adam import Adam
dataset=Cora(labeled_size=0.1,root='..\Semi_sklearn\Download\Cora')

dataset._init_dataset()
# print(dataset.data.edge_index.shape)
print(dataset.data)
#
# print(torch.sum(dataset.data.test_mask))
# print(torch.sum(dataset.data.val_mask))
# print(torch.sum(dataset.data.train_mask))
# print(torch.sum(dataset.data.labeled_mask))
# print(torch.sum(dataset.data.unlabeled_mask))

data=dataset.data
from Semi_sklearn.Alogrithm.Classifier.SDNE import SDNE
optimizer=Adam(lr=0.001)
from Semi_sklearn.Scheduler.StepLR import StepLR
LR=StepLR(step_size=10,gamma=0.9)
model=SDNE(
    input_dim=1433,
    hidden_layers=[1000,1000],
    epoch=10000,
    gamma=1e-5,
    optimizer=optimizer,
    scheduler=LR,
    device='cpu',
    weight_decay=0
)
model.fit(data)
adj=model.adjacency_matrix
print(model.embedding.shape)
#
# y_pred=model.predict(X=data.unlabeled_mask)
# y=data.y[data.unlabeled_mask]
# print(Accuracy().scoring(y,y_pred=y_pred))
# from sklearn.neighbors import KNeighborsClassifier
# model=KNeighborsClassifier()
# model.fit(adj[data.labeled_mask].numpy(),y=data.y[data.labeled_mask].numpy())
y_pred=model.predict(data.unlabeled_mask)
y=data.y[data.unlabeled_mask]
print(Accuracy().scoring(y,y_pred=y_pred))