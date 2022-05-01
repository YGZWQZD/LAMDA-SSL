from Semi_sklearn.Transform.RandomHorizontalFlip import RandomHorizontalFlip
from Semi_sklearn.Transform.RandomCrop import RandomCrop
from Semi_sklearn.Transform.RandAugment import RandAugment
from Semi_sklearn.Transform.Cutout import Cutout
from Semi_sklearn.Dataset.Vision.cifar10 import CIFAR10
from Semi_sklearn.Opitimizer.SGD import SGD
from Semi_sklearn.Scheduler.CosineAnnealingLR import CosineAnnealingLR
from Semi_sklearn.Network.WideResNet import WideResNet
from Semi_sklearn.Dataloader.TrainDataloader import TrainDataLoader
from Semi_sklearn.Dataloader.LabeledDataloader import LabeledDataLoader
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

# dataset
# dataset=CIFAR10(root='..\Download\cifar-10-python',labeled_size=4000,stratified=True,shuffle=True,download=False)
# dataset.init_dataset()
# dataset.init_transforms()

# labeled_dataset=dataset.train_dataset.get_dataset(labeled=True)
# unlabeled_dataset=dataset.train_dataset.get_dataset(labeled=False)
#
# unlabeled_X=getattr(unlabeled_dataset,'X')
# labeled_X=getattr(labeled_dataset,'X')
# labeled_y=getattr(labeled_dataset,'y')
# valid_X=getattr(dataset.test_dataset,'X')
# valid_y=getattr(dataset.test_dataset,'y')
# test_X=getattr(dataset.test_dataset,'X')
# test_y=getattr(dataset.test_dataset,'y')

# train_dataset=TrainDataset(transforms=dataset.transforms,transform=dataset.transform,
#                            target_transform=dataset.target_transform,unlabeled_transform=dataset.unlabeled_transform)
#
#
#
# valid_dataset=UnlabeledDataset(transform=dataset.valid_transform)
#
# test_dataset=UnlabeledDataset(transform=dataset.test_transform)

# augmentation

weakly_augmentation=Pipeline([('RandomHorizontalFlip',RandomHorizontalFlip()),
                              ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
                              ])

strongly_augmentation=Pipeline([('RandAugment',RandAugment(n=2,m=5,num_bins=10)),
                              ('Cutout',Cutout(v=0.5,fill=(127,127,127))),
                              ('RandomHorizontalFlip',RandomHorizontalFlip()),
                              ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
                              ])
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



# network
# network=CifarResNeXt(cardinality=4,depth=28,base_width=4,num_classes=10)
network=WideResNet(num_classes=10,depth=28,widen_factor=2,drop_rate=0)
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

# model
epoch=1
num_it_total=2**20
threshold=0.95
lambda_u=1
mu=7
T=1
weight_decay=0
device='cpu'
ema_decay=0.999




train_batch_sampler=None

valid_batch_sampler=None

test_batch_sampler=None
labeled_dataset=None
unlabeled_dataset=None
labeled_dataloader=None
unlabeled_dataloader=None
labeled_sampler=None
unlabeled_sampler=None
labeled_batch_sampler=None
unlabeled_batch_sampler=None
num_it_epoch=None
eval_epoch=None
eval_it=None
