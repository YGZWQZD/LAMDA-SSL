from LAMDA_SSL.Dataset.Graph.Cora import Cora
from LAMDA_SSL.Evaluation.Classifier.Precision import Precision
from LAMDA_SSL.Evaluation.Classifier.Recall import Recall
from LAMDA_SSL.Evaluation.Classifier.F1 import F1
from LAMDA_SSL.Evaluation.Classifier.AUC import AUC
from LAMDA_SSL.Evaluation.Classifier.Top_k_Accuracy import Top_k_Accurary
from LAMDA_SSL.Evaluation.Classifier.Confusion_Matrix import Confusion_Matrix
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Scheduler.StepLR import StepLR
from LAMDA_SSL.Opitimizer.Adam import Adam
from LAMDA_SSL.Algorithm.Classification.SDNE import SDNE

file = open("../Result/SDNE_Cora.txt", "w")

dataset=Cora(labeled_size=0.2,root='..\Download\Cora',random_state=0,default_transforms=True)
data=dataset.data
data=dataset.transform.fit_transform(data)

optimizer=Adam(lr=0.001)

scheduler= StepLR(step_size=10, gamma=0.9)

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
    epoch=500,
    eval_epoch=200,
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