from Semi_sklearn.Dataset.Text.SST2 import SST2
from Semi_sklearn.Opitimizer.SGD import SGD
from Semi_sklearn.Scheduler.CosineAnnealingLR import CosineAnnealingLR
from Semi_sklearn.Dataloader.TrainDataloader import TrainDataLoader
from Semi_sklearn.Dataloader.LabeledDataloader import LabeledDataLoader
from Semi_sklearn.Alogrithm.Classifier.Fixmatch import Fixmatch
from Semi_sklearn.Sampler.RandomSampler import RandomSampler
from Semi_sklearn.Sampler.BatchSampler import SemiBatchSampler
from Semi_sklearn.Sampler.SequentialSampler import SequentialSampler
from Semi_sklearn.Evaluation.Classification.Accuracy import Accuracy
from Semi_sklearn.Evaluation.Classification.Top_k_accuracy import Top_k_accurary
from Semi_sklearn.Evaluation.Classification.Precision import Precision
from Semi_sklearn.Evaluation.Classification.Recall import Recall
from Semi_sklearn.Evaluation.Classification.F1 import F1
from Semi_sklearn.Evaluation.Classification.AUC import AUC
from Semi_sklearn.Evaluation.Classification.Confusion_matrix import Confusion_matrix
from Semi_sklearn.Dataset.TrainDataset import TrainDataset
from Semi_sklearn.Dataset.UnlabeledDataset import UnlabeledDataset
from Semi_sklearn.Transform.TFIDF_replacement import TFIDF_replacement
from Semi_sklearn.Network.TextRCNN import TextRCNN
from Semi_sklearn.Transform.Random_swap import Random_swap

from Semi_sklearn.Transform.GloVe import Glove
vectors=Glove()
# dataset
#dataset=IMDB(root='..\Download\IMDB',stratified=True,shuffle=True,download=False,vectors=vectors,length=300)
dataset=SST2(root='..\Semi_sklearn\Download\SST2',stratified=True,shuffle=True,download=False,vectors=vectors,length=50)
dataset.init_dataset()
dataset.init_transforms()

labeled_dataset=dataset.train_dataset.get_dataset(labeled=True)
unlabeled_dataset=dataset.train_dataset.get_dataset(labeled=False)

unlabeled_X=getattr(unlabeled_dataset,'X')
labeled_X=getattr(labeled_dataset,'X')
labeled_y=getattr(labeled_dataset,'y')
valid_X=getattr(dataset.test_dataset,'X')
valid_y=getattr(dataset.test_dataset,'y')
test_X=getattr(dataset.test_dataset,'X')
test_y=getattr(dataset.test_dataset,'y')


train_dataset=TrainDataset(transforms=dataset.transforms,transform=dataset.transform,pre_transform=dataset.pre_transform,
                           target_transform=dataset.target_transform,unlabeled_transform=dataset.unlabeled_transform)

valid_dataset=UnlabeledDataset(transform=dataset.valid_transform)

test_dataset=UnlabeledDataset(transform=dataset.test_transform)

# augmentation

weakly_augmentation=Random_swap()


strongly_augmentation=TFIDF_replacement(text=labeled_X,p=0.7)



augmentation={
    'weakly_augmentation':weakly_augmentation,
    'strongly_augmentation':strongly_augmentation
}

# optimizer
optimizer=SGD(lr=0.03,momentum=0.9,nesterov=True)
scheduler=CosineAnnealingLR(eta_min=0,T_max=2**20)

#dataloader
train_dataloader=TrainDataLoader(num_workers=0)
valid_dataloader=LabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)
test_dataloader=LabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)

# sampler
train_sampler=RandomSampler(replacement=True,num_samples=64*(2**20))
train_batchsampler=SemiBatchSampler(batch_size=64,drop_last=True)
valid_sampler=SequentialSampler()
test_sampler=SequentialSampler()

# print(dataset.vectors.vec.stoi.items())

# network
# network=CifarResNeXt(cardinality=4,depth=28,base_width=4,num_classes=10)
# network=WideResNet(num_classes=10,depth=28,widen_factor=2,drop_rate=0)
#network=ResNet50(n_class=10)
network=TextRCNN(n_vocab=dataset.vectors.vec.vectors.shape[0],embedding_dim=dataset.vectors.vec.vectors.shape[1],
                 pretrained_embeddings=dataset.vectors.vec.vectors,len_seq=50,
                 num_class=2)
# evalutation
evaluation={
    'accuracy':Accuracy(),
    'top_5_accuracy':Top_k_accurary(k=5),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_matrix(normalize='true')
}

model=Fixmatch(train_dataset=train_dataset,valid_dataset=valid_dataset,test_dataset=test_dataset,
               train_dataloader=train_dataloader,valid_dataloader=valid_dataloader,test_dataloader=test_dataloader,
               augmentation=augmentation,
               network=network,epoch=1,num_it_epoch=2**20,
               num_it_total=2**20,optimizer=optimizer,scheduler=scheduler,device='cpu',
               eval_it=2000,mu=7,T=1,weight_decay=0,evaluation=evaluation,threshold=0.95,
               lambda_u=1.0,train_sampler=train_sampler,valid_sampler=valid_sampler,test_sampler=test_sampler,
               train_batch_sampler=train_batchsampler,ema_decay=0.999)

model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X,valid_X=valid_X,valid_y=valid_y)



# from sklearn.model_selection import RandomizedSearchCV
#
# model_=Fixmatch(train_dataset=train_dataset,test_dataset=test_dataset,
#                train_dataloader=train_dataloader,test_dataloader=test_dataloader,
#                augmentation=augmentation,network=network,epoch=1,num_it_epoch=2,num_it_total=2,
#                optimizer=optimizer,scheduler=scheduler,device='cpu',eval_it=1,
#                mu=7,T=1,weight_decay=0,evaluation=evaluation,train_sampler=train_sampler,
#                 test_sampler=test_sampler,train_batch_sampler=train_batchsampler,ema_decay=0.999)
#
# param_dict = {"threshold": [0.7, 1],
#               "lambda_u":[0.8,1]
#               }
#
# random_search = RandomizedSearchCV(model_, param_distributions=param_dict,
#                                    n_iter=1, cv=4,scoring='accuracy')
#
# random_search.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)

# print(labeled_X.shape)
# print(unlabeled_X.shape)





