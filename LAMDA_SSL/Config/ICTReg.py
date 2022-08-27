from LAMDA_SSL.Opitimizer.SGD import SGD
from LAMDA_SSL.Scheduler.CosineAnnealingLR import CosineAnnealingLR
from LAMDA_SSL.Dataloader.UnlabeledDataloader import UnlabeledDataLoader
from LAMDA_SSL.Dataloader.LabeledDataloader import LabeledDataLoader
from LAMDA_SSL.Sampler.RandomSampler import RandomSampler
from LAMDA_SSL.Sampler.SequentialSampler import SequentialSampler
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Transform.ToTensor import ToTensor
from LAMDA_SSL.Evaluation.Regressor.Mean_Absolute_Error import Mean_Absolute_Error
from LAMDA_SSL.Evaluation.Regressor.Mean_Squared_Error import Mean_Squared_Error
from LAMDA_SSL.Evaluation.Regressor.Mean_Squared_Log_Error import Mean_Squared_Log_Error
from LAMDA_SSL.Augmentation.Tabular.Noise import Noise


transforms = None
target_transform = None
pre_transform = None
transform = ToTensor()
unlabeled_transform = ToTensor()
test_transform = ToTensor()
valid_transform = ToTensor()

train_dataset=None
labeled_dataset=LabeledDataset(pre_transform=pre_transform,transforms=transforms,
                               transform=transform,target_transform=target_transform)

unlabeled_dataset=UnlabeledDataset(pre_transform=pre_transform,transform=unlabeled_transform)

valid_dataset=UnlabeledDataset(pre_transform=pre_transform,transform=valid_transform)

test_dataset=UnlabeledDataset(pre_transform=pre_transform,transform=test_transform)

# Batch sampler
train_batch_sampler=None
labeled_batch_sampler=None
unlabeled_batch_sampler=None
valid_batch_sampler=None
test_batch_sampler=None

# sampler
train_sampler=None
labeled_sampler=RandomSampler(replacement=True,num_samples=64*(2**20))
unlabeled_sampler=RandomSampler(replacement=True)
valid_sampler=SequentialSampler()
test_sampler=SequentialSampler()

#dataloader
train_dataloader=None
labeled_dataloader=LabeledDataLoader(batch_size=64,num_workers=0,drop_last=True)
unlabeled_dataloader=UnlabeledDataLoader(num_workers=0,drop_last=True)
valid_dataloader=UnlabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)
test_dataloader=UnlabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)

# augmentation
augmentation=Noise(noise_level=0.01)

# optimizer
optimizer=SGD(lr=0.001,momentum=0.9,nesterov=True)

# scheduler
scheduler=CosineAnnealingLR(eta_min=0,T_max=4000)

# network
network=None

evaluation={
    'Mean_Absolute_Error':Mean_Absolute_Error(),
    'Mean_Squared_Error':Mean_Squared_Error(),
    'Mean_Squared_Log_Error':Mean_Squared_Log_Error()
}

# model
weight_decay=5e-4
ema_decay=0.999
epoch=1
num_it_total=2**20
num_it_epoch=2**20
eval_epoch=None
eval_it=None
device='cpu'

parallel=None
file=None
verbose=False

alpha=0.5
dim_in=None
lambda_u=0.001
warmup=1 / 64
mu=1