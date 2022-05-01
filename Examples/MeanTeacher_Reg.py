from Semi_sklearn.Transform.RandomHorizontalFlip import RandomHorizontalFlip
from Semi_sklearn.Transform.RandomCrop import RandomCrop
from Semi_sklearn.Transform.Noise import Noise
from Semi_sklearn.Transform.Cutout import Cutout
import torch.nn as nn
from Semi_sklearn.Dataset.Table.Boston import Boston
from Semi_sklearn.Opitimizer.SGD import SGD
from Semi_sklearn.Scheduler.CosineAnnealingLR import CosineAnnealingLR
from Semi_sklearn.Network.MLP_Reg import MLP_Reg
from Semi_sklearn.Dataloader.TrainDataloader import TrainDataLoader
from Semi_sklearn.Dataloader.LabeledDataloader import LabeledDataLoader
from Semi_sklearn.Alogrithm.Regressor.MeanTeacher import MeanTeacherRegressor
from Semi_sklearn.Sampler.RandomSampler import RandomSampler
from Semi_sklearn.Sampler.BatchSampler import SemiBatchSampler
from Semi_sklearn.Sampler.SequentialSampler import SequentialSampler
from sklearn.pipeline import Pipeline
from Semi_sklearn.Evaluation.Regression.Mean_squared_error import MeanSquaredError
from Semi_sklearn.Evaluation.Regression.Mean_absolute_error import Mean_absolute_error
from Semi_sklearn.Dataset.TrainDataset import TrainDataset
from Semi_sklearn.Dataset.UnlabeledDataset import UnlabeledDataset

f = open("../Result/MeanTeacher_Reg.txt", "w")
# dataset
dataset=Boston(test_size=0.3,labeled_size=0.1,stratified=False,shuffle=True,random_state=0)
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
# print(unlabeled_X.shape)
# print(labeled_X.shape)
# print(test_X.shape)

train_dataset=TrainDataset(transforms=dataset.transforms,transform=dataset.transform,pre_transform=dataset.pre_transform,
                           target_transform=dataset.target_transform,unlabeled_transform=dataset.unlabeled_transform)

valid_dataset=UnlabeledDataset(transform=dataset.valid_transform)

test_dataset=UnlabeledDataset(transform=dataset.test_transform)

# augmentation

weakly_augmentation=Pipeline([('Noise',Noise(noise_level=0.2))
                              ])
augmentation={
    'weakly_augmentation':weakly_augmentation
}

# optimizer
optimizer=SGD(lr=0.001,momentum=0.9,nesterov=True)
scheduler=CosineAnnealingLR(eta_min=0,T_max=2000)

#dataloader
train_dataloader=TrainDataLoader(num_workers=0)
valid_dataloader=LabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)
test_dataloader=LabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)

# sampler
train_sampler=RandomSampler(replacement=True,num_samples=64*(2000))
train_batchsampler=SemiBatchSampler(batch_size=64,drop_last=True)
valid_sampler=SequentialSampler()
test_sampler=SequentialSampler()



# network
# network=CifarResNeXt(cardinality=4,depth=28,base_width=4,num_classes=10)
network=MLP_Reg(hidden_dim=[10,10],activations=[nn.ReLU(),nn.ReLU()],input_dim=labeled_X.shape[-1])

evaluation={
    'Mean_absolute_error':Mean_absolute_error(),
    'MeanSquaredError':MeanSquaredError()
}

model=MeanTeacherRegressor(train_dataset=train_dataset,valid_dataset=valid_dataset,test_dataset=test_dataset,
               train_dataloader=train_dataloader,valid_dataloader=valid_dataloader,test_dataloader=test_dataloader,
               augmentation=augmentation,network=network,epoch=1,num_it_epoch=2000,
               num_it_total=2000,optimizer=optimizer,scheduler=scheduler,device='cpu',
               eval_it=20,mu=1,weight_decay=5e-4,evaluation=evaluation,
               lambda_u=1,train_sampler=train_sampler,valid_sampler=valid_sampler,test_sampler=test_sampler,
               train_batch_sampler=train_batchsampler,ema_decay=0.999,warmup=0.4,file=f)

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





