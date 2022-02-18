from Semi_sklearn.Transform.Normalization import Normalization
from Semi_sklearn.Transform.RandomHorizontalFlip import RandomHorizontalFlip
from Semi_sklearn.Transform.RandomCrop import RandomCrop
from Semi_sklearn.Transform.RandAugment import RandAugment
from Semi_sklearn.Dataset.Vision.cifar10 import CIFAR10
from Semi_sklearn.Opitimizer.SGD import SGD
from Semi_sklearn.Scheduler.Cosine_scheduler import Cosine_schedeler
from Semi_sklearn.Network.WideResNet import WideResNet
from Semi_sklearn.Data_loader.SemiTrainDataloader import SemiTrainDataLoader
from Semi_sklearn.Data_loader.SemiTestDataloader import SemiTestDataLoader
from torch.utils.data.sampler import RandomSampler,SequentialSampler
from sklearn.pipeline import Pipeline

# dataset
dataset=CIFAR10(root='..\Download\cifar-10-python',labled_size=4000,stratified=True,shuffle=True,download=False)
dataset.init_dataset()
train_dataset=dataset.train_dataset
test_dataset=dataset.test_dataset

# augmentation
normalization=Normalization(mean=dataset.mean,std=dataset.std)
weakly_augmentation=Pipeline([('RandomHorizontalFlip',RandomHorizontalFlip()),
                              ('RandomCrop',RandomCrop(size=32,padding=int(32*0.125),padding_mode='reflect')),
                              ('Normalization',Normalization(mean=dataset.mean,std=dataset.std))
                              ])
strongly_augmentation=Pipeline([('RandomHorizontalFlip',RandomHorizontalFlip()),
                              ('RandomCrop',RandomCrop(size=32,padding=int(32*0.125),padding_mode='reflect')),
                              ('RandAugment',RandAugment(n=2,m=10)),
                              ('Normalization',Normalization(mean=dataset.mean,std=dataset.std))
                              ])
augmentation={
    'weakly_augmentation':weakly_augmentation,
    'strongly_augmentation':strongly_augmentation,
    'normalization':normalization
}

# optimizer
optimizer=SGD(lr=0.03,momentum=0.9,nesterov=True)
scheduler=Cosine_schedeler(num_warmup_steps=0,num_training_steps=2**20)

#dataloader
train_dataloader=SemiTrainDataLoader(sampler=RandomSampler,batch_size=64,num_workers=0)
test_dataloader=SemiTestDataLoader(sampler=SequentialSampler,batch_size=64,num_workers=0,drop_last=False)

# network
network=WideResNet(num_classes=10,depth=28,widen_factor=2,drop_rate=0)

# model
epoch=1
num_it_total=2**20
threshold=0.95
lambda_u=1
mu=7
T=1
weight_decay=0
device='cpu'



