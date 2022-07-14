from LAMDA_SSL.Algorithm.Classification.SSVAE import SSVAE
from LAMDA_SSL.Opitimizer.Adam import Adam
from LAMDA_SSL.Transform.ToImage import ToImage
import torch.nn as nn
from LAMDA_SSL.Dataloader.UnlabeledDataloader import UnlabeledDataLoader
from LAMDA_SSL.Dataloader.LabeledDataloader import LabeledDataLoader
from LAMDA_SSL.Sampler.RandomSampler import RandomSampler
from LAMDA_SSL.Sampler.SequentialSampler import SequentialSampler
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Evaluation.Classifier.Top_k_Accuracy import Top_k_Accurary
from LAMDA_SSL.Evaluation.Classifier.Precision import Precision
from LAMDA_SSL.Evaluation.Classifier.Recall import Recall
from LAMDA_SSL.Evaluation.Classifier.F1 import F1
from LAMDA_SSL.Evaluation.Classifier.AUC import AUC
from LAMDA_SSL.Evaluation.Classifier.Confusion_Matrix import Confusion_Matrix
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Dataset.Vision.Mnist import Mnist

# dataset
dataset=Mnist(root='..\Download\mnist',labeled_size=6000,stratified=True,shuffle=True,download=False,random_state=0,default_transforms=True)

labeled_X=dataset.labeled_X
labeled_y=dataset.labeled_y

unlabeled_X=dataset.unlabeled_X

test_X=dataset.test_X
test_y=dataset.test_y

valid_X=dataset.valid_X
valid_y=dataset.valid_y

labeled_dataset=LabeledDataset(pre_transform=dataset.pre_transform,transforms=dataset.transforms,
                               transform=dataset.transform,target_transform=dataset.target_transform)
unlabeled_dataset=UnlabeledDataset(pre_transform=dataset.pre_transform,transform=dataset.unlabeled_transform)
valid_dataset=UnlabeledDataset(pre_transform=dataset.pre_transform,transform=dataset.valid_transform)
test_dataset=UnlabeledDataset(pre_transform=dataset.pre_transform,transform=dataset.test_transform)

#dataloader
labeled_dataloader=LabeledDataLoader(batch_size=100,num_workers=0,drop_last=True)
unlabeled_dataloader=UnlabeledDataLoader(num_workers=0,drop_last=True)
valid_dataloader=UnlabeledDataLoader(batch_size=100,num_workers=0,drop_last=False)
test_dataloader=UnlabeledDataLoader(batch_size=100,num_workers=0,drop_last=False)

# sampler
labeled_sampler=RandomSampler(replacement=True,num_samples=100*540)
unlabeled_sampler=RandomSampler(replacement=False)
test_sampler=SequentialSampler()
valid_sampler=SequentialSampler()

# optimizer
optimizer=Adam(lr=3e-4)

# evalutation
evaluation={
    'Accuracy':Accuracy(),
    'Top_5_Accuracy':Top_k_Accurary(k=5),
    'Precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_Matrix':Confusion_Matrix(normalize='true')
}

file = open("../Result/SSVAE_MNIST.txt", "w")

model=SSVAE(alpha=5,num_labeled=6000,dim_in=(28,28),num_classes=10,
             dim_z=100,
             dim_hidden_de=[500, 500],
             dim_hidden_en_y=[500, 500], dim_hidden_en_z=[500, 500],
             activations_de=[nn.Softplus(), nn.Softplus()],
             activations_en_y=[nn.Softplus(), nn.Softplus()],
             activations_en_z=[nn.Softplus(), nn.Softplus()],
             mu=1,weight_decay=0,epoch=100,num_it_epoch=540,num_it_total=540*100,eval_it=2000,device='cpu',
             labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset,valid_dataset=valid_dataset,test_dataset=test_dataset,
             labeled_sampler=labeled_sampler,unlabeled_sampler=unlabeled_sampler,valid_sampler=valid_sampler,test_sampler=test_sampler,
             labeled_dataloader=labeled_dataloader,unlabeled_dataloader=unlabeled_dataloader,valid_dataloader=valid_dataloader,test_dataloader=test_dataloader,
             optimizer=optimizer,
             evaluation=evaluation,
             file=file,verbose=True)

model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X,valid_X=valid_X,valid_y=valid_y)

performance=model.evaluate(X=test_X,y=test_y)

result=model.y_pred

print(result,file=file)

print(performance,file=file)

X=model.generate(100)

for _ in range(100):
    img=ToImage()(X[_]*256)
    img.convert('RGB').save('../Result/Imgs/SSVAE/' + str(_) + '.jpg')
