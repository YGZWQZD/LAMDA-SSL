from lamda_ssl.Transform.RandomHorizontalFlip import RandomHorizontalFlip
from lamda_ssl.Transform.RandomCrop import RandomCrop
from lamda_ssl.Transform.RandAugment import RandAugment
from lamda_ssl.Transform.Cutout import Cutout
from lamda_ssl.Algorithm.Classifier.SSVAE import SSVAE
from lamda_ssl.Dataset.Vision.cifar10 import CIFAR10
from lamda_ssl.Opitimizer.Adam import Adam
from lamda_ssl.Transform.ToImage import ToImage
import torch.nn as nn
from lamda_ssl.Scheduler.CosineAnnealingLR import CosineAnnealingLR
from lamda_ssl.Network.WideResNet import WideResNet
from lamda_ssl.Dataloader.TrainDataloader import TrainDataLoader
from lamda_ssl.Dataloader.LabeledDataloader import LabeledDataLoader
from lamda_ssl.Algorithm.Classifier.MeanTeacher import MeanTeacherClassifier
from lamda_ssl.Sampler.RandomSampler import RandomSampler
from lamda_ssl.Sampler.BatchSampler import SemiBatchSampler
from lamda_ssl.Sampler.SequentialSampler import SequentialSampler
from sklearn.pipeline import Pipeline
from lamda_ssl.Evaluation.Classification.Accuracy import Accuracy
from lamda_ssl.Evaluation.Classification.Top_k_accuracy import Top_k_accurary
from lamda_ssl.Evaluation.Classification.Precision import Precision
from lamda_ssl.Evaluation.Classification.Recall import Recall
from lamda_ssl.Evaluation.Classification.F1 import F1
from lamda_ssl.Evaluation.Classification.AUC import AUC
from lamda_ssl.Evaluation.Classification.Confusion_matrix import Confusion_matrix
from lamda_ssl.Dataset.TrainDataset import TrainDataset
from lamda_ssl.Dataset.UnlabeledDataset import UnlabeledDataset
from lamda_ssl.Transform.ImageToTensor import ToTensor
from lamda_ssl.Scheduler.Linear_warmup import Linear_warmup

# dataset
from lamda_ssl.Dataset.Vision.Mnist import Mnist
dataset=Mnist(root='..\Semi_sklearn\Download\mnist',labeled_size=3000,stratified=True,shuffle=True,download=False,random_state=0)
# f = open("../Result/SSVAE.txt", "w")
# dataset
# dataset=CIFAR10(root='..\Download\cifar-10-python',labeled_size=4000,stratified=True,shuffle=True,download=False)

labeled_dataset=dataset.train_dataset.get_dataset(labeled=True)
unlabeled_dataset=dataset.train_dataset.get_dataset(labeled=False)

unlabeled_X=getattr(unlabeled_dataset,'X')
labeled_X=getattr(labeled_dataset,'X')
labeled_y=getattr(labeled_dataset,'y')
valid_X=getattr(dataset.test_dataset,'X')
valid_y=getattr(dataset.test_dataset,'y')
test_X=getattr(dataset.test_dataset,'X')
test_y=getattr(dataset.test_dataset,'y')
# print(labeled_X.shape)
# print(unlabeled_X.shape)

train_dataset=TrainDataset(transforms=dataset.transforms,transform=dataset.transform,pre_transform=dataset.pre_transform,
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
optimizer=Adam(lr=3e-4)
# scheduler=Linear_warmup(num_warmup_steps=15,num_training_steps=10)

#dataloader
train_dataloader=TrainDataLoader(num_workers=0)
valid_dataloader=LabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)
test_dataloader=LabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)

# sampler
train_sampler=[RandomSampler(replacement=True,num_samples=100*570),RandomSampler(replacement=False)]
train_batchsampler=SemiBatchSampler(batch_size=100,drop_last=True)
test_sampler=SequentialSampler()
valid_sampler=SequentialSampler()


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


model=SSVAE(dim_in=(1,28,28),num_class=10,
                     dim_z=50,
                     dim_hidden_de=[500, 500],
                     dim_hidden_en_y=[500, 500], dim_hidden_en_z=[500, 500],
                     activations_de=[nn.Softplus(), nn.Softplus()],
                     activations_en_y=[nn.Softplus(), nn.Softplus()],
                     activations_en_z=[nn.Softplus(), nn.Softplus()],
                     alpha=0.1,
                     num_labeled=3000,
                     train_dataset=train_dataset,valid_dataset=valid_dataset,test_dataset=test_dataset,
                     train_dataloader=train_dataloader,valid_dataloader=valid_dataloader,test_dataloader=test_dataloader,
                     epoch=0,num_it_epoch=570,
                     num_it_total=570*0,optimizer=optimizer,device='cpu',
                     eval_it=1000,mu=1,weight_decay=0,evaluation=evaluation,
                     train_sampler=train_sampler,valid_sampler=valid_sampler,test_sampler=test_sampler,
                     train_batch_sampler=train_batchsampler,file=None)

model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X,valid_X=valid_X,valid_y=valid_y)
X=model.generate(100)
# print(X.shape)
X=X.detach().numpy()
print(X.shape)
for _ in range(100):
    img=ToImage()(X[_].squeeze(0))
    img.convert('RGB').save('../Result/Imgs/SSVAE/' + str(_) + '.jpg')
    # print(type(img))
    # import matplotlib.pyplot as plt

    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()


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





