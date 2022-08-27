from LAMDA_SSL.Dataset.Text.SST2 import SST2
from LAMDA_SSL.Opitimizer.SGD import SGD
from LAMDA_SSL.Scheduler.CosineAnnealingLR import CosineAnnealingLR
from LAMDA_SSL.Dataloader.UnlabeledDataloader import UnlabeledDataLoader
from LAMDA_SSL.Dataloader.LabeledDataloader import LabeledDataLoader
from LAMDA_SSL.Algorithm.Classification.FlexMatch import FlexMatch
from LAMDA_SSL.Sampler.RandomSampler import RandomSampler
from LAMDA_SSL.Sampler.SequentialSampler import SequentialSampler
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Evaluation.Classifier.Precision import Precision
from LAMDA_SSL.Evaluation.Classifier.Recall import Recall
from LAMDA_SSL.Evaluation.Classifier.F1 import F1
from LAMDA_SSL.Evaluation.Classifier.AUC import AUC
from LAMDA_SSL.Evaluation.Classifier.Confusion_Matrix import Confusion_Matrix
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Augmentation.Text.TFIDFReplacement import TFIDFReplacement
from LAMDA_SSL.Augmentation.Text.RandomSwap import RandomSwap
from LAMDA_SSL.Network.TextRCNN import TextRCNN
from LAMDA_SSL.Transform.Text.GloVe import Glove

# dataset
dataset=SST2(root='..\Download\SST2',stratified=True,shuffle=True,download=False,vectors=Glove(cache='..\Download\Glove\.vector_cache'),length=50,default_transforms=True)

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

# sampler
labeled_sampler=RandomSampler(replacement=True,num_samples=64*(2**20))
unlabeled_sampler=RandomSampler(replacement=True)
valid_sampler=SequentialSampler()
test_sampler=SequentialSampler()

#dataloader
labeled_dataloader=LabeledDataLoader(batch_size=64,num_workers=0,drop_last=True)
unlabeled_dataloader=UnlabeledDataLoader(num_workers=0,drop_last=True)
valid_dataloader=UnlabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)
test_dataloader=UnlabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)

# augmentation

weak_augmentation=RandomSwap(n=1)

strong_augmentation=TFIDFReplacement(text=labeled_X,p=0.7)

augmentation={
    'weak_augmentation':weak_augmentation,
    'strong_augmentation':strong_augmentation
}

# optimizer
optimizer=SGD(lr=0.03,momentum=0.9,nesterov=True)

# scheduler
scheduler=CosineAnnealingLR(eta_min=0,T_max=2**20)

# network
network=TextRCNN(n_vocab=dataset.vectors.vec.vectors.shape[0],embedding_dim=dataset.vectors.vec.vectors.shape[1],
                 pretrained_embeddings=dataset.vectors.vec.vectors,len_seq=50,
                 num_classes=2)
# evalutation
evaluation={
    'accuracy':Accuracy(),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_Matrix(normalize='true')
}

file = open("../Result/FixMatch_SST2.txt", "w")

model=FlexMatch(threshold=0.95,lambda_u=1.0,T=1.0,mu=7,ema_decay=0.999,weight_decay=5e-4,
               epoch=1,num_it_epoch=2**20,num_it_total=2**20,eval_it=2000,device='cpu',
               labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
               valid_dataset=valid_dataset, test_dataset=test_dataset,
               labeled_sampler=labeled_sampler,unlabeled_sampler=unlabeled_sampler,
               valid_sampler=valid_sampler,test_sampler=test_sampler,
               labeled_dataloader=labeled_dataloader, unlabeled_dataloader=unlabeled_dataloader,
               valid_dataloader=valid_dataloader, test_dataloader=test_dataloader,
               augmentation=augmentation,network=network,optimizer=optimizer,scheduler=scheduler,
               evaluation=evaluation,verbose=True,file=file)

model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X,valid_X=valid_X,valid_y=valid_y)

performance=model.evaluate(X=test_X,y=test_y)

result=model.y_pred

print(result,file=file)

print(performance,file=file)