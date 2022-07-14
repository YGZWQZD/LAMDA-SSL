from LAMDA_SSL.Augmentation.Vision.RandomHorizontalFlip import RandomHorizontalFlip
from LAMDA_SSL.Augmentation.Vision.RandomCrop import RandomCrop
from LAMDA_SSL.Opitimizer.SGD import SGD
from LAMDA_SSL.Scheduler.CosineAnnealingLR import CosineAnnealingLR
from LAMDA_SSL.Network.WideResNet import WideResNet
from LAMDA_SSL.Dataloader.UnlabeledDataloader import UnlabeledDataLoader
from LAMDA_SSL.Dataloader.LabeledDataloader import LabeledDataLoader
from LAMDA_SSL.Sampler.RandomSampler import RandomSampler
from LAMDA_SSL.Sampler.SequentialSampler import SequentialSampler
from sklearn.pipeline import Pipeline
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Evaluation.Classifier.Top_k_Accuracy import Top_k_Accurary
from LAMDA_SSL.Evaluation.Classifier.Precision import Precision
from LAMDA_SSL.Evaluation.Classifier.Recall import Recall
from LAMDA_SSL.Evaluation.Classifier.F1 import F1
from LAMDA_SSL.Evaluation.Classifier.AUC import AUC
from LAMDA_SSL.Evaluation.Classifier.Confusion_Matrix import Confusion_Matrix
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Transform.Vision.Normalization import Normalization
from LAMDA_SSL.Transform.ToTensor import ToTensor
from LAMDA_SSL.Transform.ToImage import ToImage

mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

pre_transform = ToImage()
transforms = None
target_transform = None
transform = Pipeline([('ToTensor', ToTensor(dtype='float',image=True)),
                    ('Normalization', Normalization(mean=mean, std=std))
                    ])
unlabeled_transform = Pipeline([('ToTensor', ToTensor(dtype='float',image=True)),
                                ('Normalization', Normalization(mean=mean, std=std))
                                ])
test_transform = Pipeline([('ToTensor', ToTensor(dtype='float',image=True)),
                                ('Normalization', Normalization(mean=mean, std=std))
                                ])
valid_transform = Pipeline([('ToTensor', ToTensor(dtype='float',image=True)),
                                 ('Normalization', Normalization(mean=mean, std=std))
                                 ])

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

# network
network=WideResNet(num_classes=10,depth=28,widen_factor=2,drop_rate=0)

# optimizer
optimizer=SGD(lr=0.03,momentum=0.9,nesterov=True)

# scheduler
scheduler=CosineAnnealingLR(eta_min=0,T_max=2**20)

# augmentation
augmentation=Pipeline([('RandomHorizontalFlip',RandomHorizontalFlip()),
                        ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
                      ])

# evalutation
evaluation={
    'accuracy':Accuracy(),
    'top_5_accuracy':Top_k_Accurary(k=5),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_Matrix(normalize='true')
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
lambda_u=100
num_classes=None
T=0.5
warmup=1 / 64
mu=1