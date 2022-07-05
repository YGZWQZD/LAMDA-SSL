from lamda_ssl.Dataset.Graph.Cora import Cora
from lamda_ssl.Evaluation.Classification.Precision import Precision
from lamda_ssl.Evaluation.Classification.Recall import Recall
from lamda_ssl.Evaluation.Classification.F1 import F1
from lamda_ssl.Evaluation.Classification.AUC import AUC
from lamda_ssl.Evaluation.Classification.Top_k_Accuracy import Top_k_Accurary
from lamda_ssl.Evaluation.Classification.Confusion_Matrix import Confusion_Matrix
from lamda_ssl.Evaluation.Classification.Accuracy import Accuracy
from lamda_ssl.Scheduler.StepLR import StepLR
from lamda_ssl.Opitimizer.Adam import Adam
from lamda_ssl.Algorithm.Classifier.SDNE import SDNE

file = open("../Result/SDNE_Cora.txt", "w")

dataset=Cora(labeled_size=0.2,root='..\Download\Cora',random_state=0,default_transforms=True)
data=dataset.transform.fit_transform(dataset.data)

optimizer=Adam(lr=0.001)

scheduler=StepLR(step_size=10,gamma=0.9)

evaluation={
    'accuracy':Accuracy(),
    'top_5_accuracy':Top_k_Accurary(k=5),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_Matrix(normalize='true')
}

model=SDNE(
    hidden_layers=[1000,1000],
    gamma=1e-5,
    alpha=1e-3,
    beta=10,
    epoch=1000,
    eval_epoch=100,
    weight_decay=0,
    device='cpu',
    optimizer=optimizer,
    scheduler=scheduler,
    evaluation=evaluation,
    verbose=True,
    file=file
)

model.fit(data,valid_X=data.val_mask)

performance=model.evaluate(X=data.test_mask)

result=model.y_pred

print(result,file=file)

print(performance,file=file)