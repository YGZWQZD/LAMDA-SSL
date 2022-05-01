from Semi_sklearn.Dataset.Graph.Cora import Cora
#
import numpy as np
import torch
from Semi_sklearn.Evaluation.Classification.Accuracy import Accuracy
from Semi_sklearn.Evaluation.Classification.Top_k_accuracy import Top_k_accurary
from Semi_sklearn.Evaluation.Classification.Precision import Precision
from Semi_sklearn.Evaluation.Classification.Recall import Recall
from Semi_sklearn.Evaluation.Classification.F1 import F1
from Semi_sklearn.Evaluation.Classification.AUC import AUC
from Semi_sklearn.Evaluation.Classification.Confusion_matrix import Confusion_matrix
from Semi_sklearn.Opitimizer.Adam import Adam
f = open("../Result/GCN.txt", "w")
dataset=Cora(labeled_size=0.2,root='..\Semi_sklearn\Download\Cora',random_state=0)

dataset.init_dataset()
dataset.init_transforms()
# print(dataset.data.edge_index.shape)
#
# print(torch.sum(dataset.data.test_mask))
# print(torch.sum(dataset.data.val_mask))
# print(torch.sum(dataset.data.train_mask))
# print(torch.sum(dataset.data.labeled_mask))
# print(torch.sum(dataset.data.unlabeled_mask))
evaluation={
    'accuracy':Accuracy(),
    'top_5_accuracy':Top_k_accurary(k=5),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_matrix(normalize='true')
}
data=dataset.data
from Semi_sklearn.Alogrithm.Classifier.GCN import GCN
optimizer=Adam(lr=0.02)

model=GCN(
    num_features=1433,
    num_classes=7,
    normalize=True,
    epoch=1000,
    eval_epoch=100,
    optimizer=optimizer,
    scheduler=None,
    device='cpu',
    evaluation=evaluation,
    weight_decay=0.05,
    file=f
)
model.fit(data,valid_X=data.unlabeled_mask)