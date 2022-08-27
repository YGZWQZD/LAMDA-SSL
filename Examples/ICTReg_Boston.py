from LAMDA_SSL.Augmentation.Tabular.Noise import Noise
import torch.nn as nn
from LAMDA_SSL.Dataset.Tabular.Boston import Boston
from LAMDA_SSL.Opitimizer.SGD import SGD
from LAMDA_SSL.Scheduler.CosineAnnealingLR import CosineAnnealingLR
from LAMDA_SSL.Network.MLPReg import MLPReg
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Dataloader.LabeledDataloader import LabeledDataLoader
from LAMDA_SSL.Dataloader.UnlabeledDataloader import UnlabeledDataLoader
from LAMDA_SSL.Algorithm.Regression.ICTReg import ICTReg
from LAMDA_SSL.Sampler.RandomSampler import RandomSampler
from LAMDA_SSL.Sampler.SequentialSampler import SequentialSampler
from LAMDA_SSL.Evaluation.Regressor.Mean_Absolute_Error import Mean_Absolute_Error
from LAMDA_SSL.Evaluation.Regressor.Mean_Squared_Error import Mean_Squared_Error
from LAMDA_SSL.Evaluation.Regressor.Mean_Squared_Log_Error import Mean_Squared_Log_Error
import numpy as np

# dataset
dataset=Boston(test_size=0.3,labeled_size=0.1,stratified=False,shuffle=True,random_state=0,default_transforms=True)

labeled_X=dataset.labeled_X
labeled_y=dataset.labeled_y

unlabeled_X=dataset.unlabeled_X

test_X=dataset.test_X
test_y=dataset.test_y

valid_X=dataset.valid_X
valid_y=dataset.valid_y

# Pre_transform
pre_transform=dataset.pre_transform
pre_transform.fit(np.vstack([labeled_X, unlabeled_X]))

labeled_X=pre_transform.transform(labeled_X)
unlabeled_X=pre_transform.transform(unlabeled_X)
test_X=pre_transform.transform(test_X)

labeled_dataset=LabeledDataset(transforms=dataset.transforms,
                               transform=dataset.transform,target_transform=dataset.target_transform)

unlabeled_dataset=UnlabeledDataset(transform=dataset.unlabeled_transform)

valid_dataset=UnlabeledDataset(transform=dataset.valid_transform)

test_dataset=UnlabeledDataset(transform=dataset.test_transform)

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

augmentation=Noise(noise_level=0.01)

# optimizer
optimizer=SGD(lr=0.001,momentum=0.9,nesterov=True)
scheduler=CosineAnnealingLR(eta_min=0,T_max=4000)

# network
network=MLPReg(hidden_dim=[100,50,10],activations=[nn.ReLU(),nn.ReLU(),nn.ReLU()],dim_in=labeled_X.shape[-1])

evaluation={
    'Mean_Absolute_Error':Mean_Absolute_Error(),
    'Mean_Squared_Error':Mean_Squared_Error(),
    'Mean_Squared_Log_Error':Mean_Squared_Log_Error()
}

file = open("../Result/ICTReg_Boston.txt", "w")

model=ICTReg(alpha=0.5,lambda_u=0.001,warmup=1/64,
               mu=1,weight_decay=5e-4,ema_decay=0.999,
               epoch=1,num_it_epoch=4000,
               num_it_total=4000,
               eval_it=200,device='cpu',
               labeled_dataset=labeled_dataset,
               unlabeled_dataset=unlabeled_dataset,
               valid_dataset=valid_dataset,
               test_dataset=test_dataset,
               labeled_sampler=labeled_sampler,
               unlabeled_sampler=unlabeled_sampler,
               valid_sampler=valid_sampler,
               test_sampler=test_sampler,
               labeled_dataloader=labeled_dataloader,
               unlabeled_dataloader=unlabeled_dataloader,
               valid_dataloader=valid_dataloader,
               test_dataloader=test_dataloader,
               augmentation=augmentation,network=network,
               optimizer=optimizer,scheduler=scheduler,
               evaluation=evaluation,file=file,verbose=True)

model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X,valid_X=valid_X,valid_y=valid_y)

performance=model.evaluate(X=test_X,y=test_y)

result=model.y_pred

print(result,file=file)

print(performance,file=file)