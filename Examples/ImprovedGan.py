import torch

from Semi_sklearn.Transform.RandomHorizontalFlip import RandomHorizontalFlip
from Semi_sklearn.Transform.RandomCrop import RandomCrop
from Semi_sklearn.Transform.RandAugment import RandAugment
from Semi_sklearn.Transform.Cutout import Cutout
from Semi_sklearn.Model.Classifier.SSVAE import SSVAE
from Semi_sklearn.Dataset.Vision.cifar10 import CIFAR10
from Semi_sklearn.Opitimizer.Adam import Adam
from Semi_sklearn.Transform.ToImage import ToImage
from Semi_sklearn.Scheduler.CosineAnnealingLR import CosineAnnealingLR
from Semi_sklearn.Network.WideResNet import WideResNet
from Semi_sklearn.Dataloader.TrainDataloader import TrainDataLoader
from Semi_sklearn.Dataloader.LabeledDataloader import LabeledDataLoader
from Semi_sklearn.Model.Classifier.MeanTeacher import MeanTeacherClassifier
from Semi_sklearn.Sampler.RandomSampler import RandomSampler
from Semi_sklearn.Sampler.BatchSampler import SemiBatchSampler
from Semi_sklearn.Sampler.SequentialSampler import SequentialSampler
from sklearn.pipeline import Pipeline
from Semi_sklearn.Evaluation.Classification.Accuracy import Accuracy
from Semi_sklearn.Evaluation.Classification.Top_k_accuracy import Top_k_accurary
from Semi_sklearn.Evaluation.Classification.Precision import Precision
from Semi_sklearn.Evaluation.Classification.Recall import Recall
from Semi_sklearn.Evaluation.Classification.F1 import F1
from Semi_sklearn.Evaluation.Classification.AUC import AUC
from Semi_sklearn.Evaluation.Classification.Confusion_matrix import Confusion_matrix
from Semi_sklearn.Dataset.TrainDataset import TrainDataset
from Semi_sklearn.Dataset.UnlabeledDataset import UnlabeledDataset
from Semi_sklearn.Transform.ImageToTensor import ToTensor
from Semi_sklearn.Scheduler.Linear_warmup import Linear_warmup
from Semi_sklearn.Model.Classifier.ImprovedGan import ImprovedGan

from Semi_sklearn.Dataset.Vision.Mnist import Mnist
dataset=Mnist(root='..\Download\mnist',stratified=True,shuffle=True,download=False)
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

train_dataset=TrainDataset(transforms=dataset.transforms,transform=dataset.transform,
                           target_transform=dataset.target_transform,unlabeled_transform=dataset.unlabeled_transform)

valid_dataset=UnlabeledDataset(transform=dataset.valid_transform)

test_dataset=UnlabeledDataset(transform=dataset.test_transform)

# augmentation

# weakly_augmentation=Pipeline([('RandomHorizontalFlip',RandomHorizontalFlip()),
#                               ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
#                               ])
#
# strongly_augmentation=Pipeline([('RandAugment',RandAugment(n=2,m=10,num_bins=30)),
#                               ('Cutout',Cutout(v=0.5,fill=(127,127,127))),
#                               ('RandomHorizontalFlip',RandomHorizontalFlip()),
#                               ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
#                               ])
# augmentation={
#     'weakly_augmentation':weakly_augmentation,
#     'strongly_augmentation':strongly_augmentation
# }
# optimizer
optimizer=Adam(lr=0.02)
# scheduler=Linear_warmup(num_warmup_steps=15,num_training_steps=10)

#dataloader
train_dataloader=TrainDataLoader(num_workers=0)
valid_dataloader=LabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)
test_dataloader=LabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)

# sampler
train_sampler=[RandomSampler(replacement=True,num_samples=540*100),RandomSampler(replacement=False)]
train_batchsampler=SemiBatchSampler(batch_size=100,drop_last=True)
test_sampler=SequentialSampler()
valid_sampler=SequentialSampler()

# weakly_augmentation=Pipeline([('RandomHorizontalFlip',RandomHorizontalFlip()),
#                               ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
#                               ])

# network
# network=CifarResNeXt(cardinality=4,depth=28,base_width=4,num_classes=10)
# network=WideResNet(num_classes=10,depth=28,widen_factor=2,drop_rate=0)
#network=ResNet50(n_class=10)


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


model=ImprovedGan(dim_in=28*28,num_class=10,dim_z=500,dim_hidden=500,
                     train_dataset=train_dataset,valid_dataset=valid_dataset,test_dataset=test_dataset,
                     train_dataloader=train_dataloader,valid_dataloader=valid_dataloader,test_dataloader=test_dataloader,
                     epoch=1000,num_it_epoch=540,
                     num_it_total=540*1000,optimizer=optimizer,device='cpu',
                     eval_it=200,mu=1,weight_decay=5e-4,evaluation=evaluation,
                     train_sampler=train_sampler,valid_sampler=valid_sampler,test_sampler=test_sampler,
                     train_batch_sampler=train_batchsampler,lambda_u=1)

model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X,valid_X=valid_X,valid_y=valid_y)

X=model.generate(100)
print(X.shape)
X=X.view(X.shape[0],28,28)
X=X.detach().numpy()
img=ToImage()(X[0])
import matplotlib.pyplot as plt
plt.imshow(img)
plt.axis('off')
plt.show()

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





