from LAMDA_SSL.Dataset.Graph.Cora import Cora
from LAMDA_SSL.Evaluation.Classification.Accuracy import Accuracy
from LAMDA_SSL.Evaluation.Classification.Top_k_Accuracy import Top_k_Accurary
from LAMDA_SSL.Evaluation.Classification.Precision import Precision
from LAMDA_SSL.Evaluation.Classification.Recall import Recall
from LAMDA_SSL.Evaluation.Classification.F1 import F1
from LAMDA_SSL.Evaluation.Classification.AUC import AUC
from LAMDA_SSL.Evaluation.Classification.Confusion_Matrix import Confusion_Matrix
from LAMDA_SSL.Opitimizer.Adam import Adam
from LAMDA_SSL.Algorithm.Classifier.GCN import GCN

file = open("../Result/GCN_Cora.txt", "w")

dataset=Cora(labeled_size=0.2,root='..\Download\Cora',random_state=0,default_transforms=True)
data=dataset.transform.fit_transform(dataset.data)

evaluation={
    'accuracy':Accuracy(),
    'top_5_accuracy':Top_k_Accurary(k=5),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_Matrix(normalize='true')
}

optimizer=Adam(lr=0.01)

model=GCN(
    num_features=1433,
    normalize=True,
    epoch=2000,
    eval_epoch=100,
    weight_decay=5e-4,
    device='cpu',
    optimizer=optimizer,
    evaluation=evaluation,
    file=file,
    verbose=True
)

model.fit(data,valid_X=data.val_mask)

performance=model.evaluate(X=data.test_mask)

result=model.y_pred

print(result,file=file)

print(performance,file=file)