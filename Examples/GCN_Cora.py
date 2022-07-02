from lamda_ssl.Dataset.Graph.Cora import Cora
from lamda_ssl.Evaluation.Classification.Accuracy import Accuracy
from lamda_ssl.Evaluation.Classification.Top_k_Accuracy import Top_k_Accurary
from lamda_ssl.Evaluation.Classification.Precision import Precision
from lamda_ssl.Evaluation.Classification.Recall import Recall
from lamda_ssl.Evaluation.Classification.F1 import F1
from lamda_ssl.Evaluation.Classification.AUC import AUC
from lamda_ssl.Evaluation.Classification.Confusion_Matrix import Confusion_Matrix
from lamda_ssl.Opitimizer.Adam import Adam
from lamda_ssl.Algorithm.Classifier.GCN import GCN

file = open("../Result/GCN_Cora.txt", "w")

dataset=Cora(labeled_size=0.2,root='..\lamda_ssl\Download\Cora',random_state=0,default_transforms=True)
data=dataset.data

evaluation={
    'accuracy':Accuracy(),
    'top_5_accuracy':Top_k_Accurary(k=5),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_Matrix(normalize='true')
}

optimizer=Adam(lr=0.02)

model=GCN(
    num_features=1433,
    normalize=True,
    epoch=1000,
    eval_epoch=100,
    weight_decay=0.05,
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