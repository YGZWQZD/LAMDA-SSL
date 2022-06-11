#  介绍

Semi-sklearn是一个有效易用的半监督学习工具包。目前该工具包包含30种半监督学习算法，其中基于传统机器学习模型的算法13种，基于深度神经网络模型的算法17种，可用于处理结构化数据、图像数据、文本数据、图结构数据4种数据类型，可用于分类、回归、聚类3种任务，包含数据管理、数据变换、算法应用、模型评估等多个模块，便于实现端到端的半监督学习过程，兼容目前主流的机器学习工具包scikit-learn和深度学习工具包pytorch，具备完善的功能，标准的接口和详尽的文档。


##  设计思想

Semi-sklearn的整体设计思想如图所示。Semi-sklearn参考了sklearn工具包的底层实现，所有算法都使用了与sklearn相似的接口。 在sklearn中的学习器都继承了Estimator这一父类，Estimator表示一个估计器，利用现有数据建立模型对未来的数据做出预测，对估计器存在fit()和transform()两个方法，其中fit()方法是一个适配过程，即利用现有数据建立模型，对应了机器学习中的训练过程，transform()方法是一个转换过程，即利用fit()过后的模型对新数据进行预测。

Semi-sklearn中的预测器通过继承半监督预测器类SemiEstimator间接继承了sklearn中的Estimator。由于sklearn中fit()方法使用的数据往往包含样本和标注两项，在半监督学习中，模型的训练过程中同时使用有标注数据、标注和无标注数据，因此Estimator的fit()方法不方便直接用于半监督学习算法。虽然sklearn中也实现了自训练方法和基于图的方法两类半监督学习算法，它们也继承了Estimator类，但是为了使用fit()方法的接口，sklearn将有标注样本与无标注数据样本结合在一起作为fit()的样本输入，将标注输入中无标注数据对应的标注记为-1，这种处理方式虽然可以适应Estimator的接口，但是也存在局限性，尤其使在一些二分类场景下往往用-1表示有标注数据的负例标注，与无标注数据会发生冲突，因此针对半监督学习在Estimator的基础上重新建立新类SemiEstimator具有必要性，SemiEstimator的fit()方法包含有标注数据、标注和无标注数据三部分输入，更好地契合了半监督学习的应用场景，避免了要求用户自己对数据进行组合处理，也避免了无标注数据与二分类负类的冲突，相较Estimator使用起来更加方便。

半监督学习一般分为归纳式学习和直推式学习，区别在于是否直接使用待预测数据作为训练过程中的无标注数据。Semi-sklearn中使用两个类InductiveEstimator和Transductive分别对应了归纳式学习和直推式学习两类半监督学习方法，均继承了SemiEstimator类。

在sklearn中，为了使估计器针对不同的任务可以具备相应的功能，sklearn针对估计器的不同使用场景开发了与场景对应的组件（Mixin），sklearn中的估计器往往会同时继承Estimator和相应组件，从而使估计器同时拥有基本的适配和预测功能，还能拥有不同组件对应的处理不同任务场景的功能。其中关键组件包括用于分类任务的ClassifierMixin、用于回归任务的RegressorMixin、用于聚类任务的ClusterMixin和用于数据转换的TransformerMixin，在Semi-sklearn中同样使用了这些组件。

另外，不同于经典机器学习中常用的sklearn框架，深度学习在经常使用pytorch框架，pytorch各组件间存在较大的依赖关系（如图3-2所示），耦合度高，例如数据集（Dataset）与数据加载器（Dataloader）的耦合、优化器（Optimizer）和调度器（Scheduler）的耦合、采样器（Sampler）与批采样器（BatchSampler）的耦合等，没有像sklearn一样的简单的逻辑和接口，对用户自身要求较高，较不方便，为在同一工具包在同时包含经典机器学习方法和深度学习方法造成了较大困难，为了解决经典机器学习方法和深度学习方法难以融合于相同框架的问题，Semi-sklearn用DeepModelMixin这一组件使基于pytorch开发的深度半监督模型拥有了与经典机器学习方法相同接口和使用方式，Semi-sklearn中的深度半监督学习算法都继承了这一组件。DeepModelMixin对pytorch各模块进行了解耦，便于用户独立更换深度学习中数据加载器、网络结构、优化器等模块，而不需要考虑更换对其他模块造成的影响，DeepModelMixin会自动处理这些影响，使用户可以像调用经典的半监督学习算法一样便捷地调用深度半监督学习算法。

## 数据管理

Semi-sklearn拥有强大的数据管理和数据处理功能。在Semi-sklearn中，一个半监督数据集整体可以用一个SemiDataset类进行管理，SemiDataset类可以同时管理TrainDataset、ValidDataset、TestDataset三个子数据集，分别对应了机器学习任务中的训练数据集、验证数据集和测试数据集，在最底层数据集分为LabeledDataset和UnlabeledDataset两类，分别对应了半监督学习中的有标注数据与无标注数据，训练集往往同时包含有标注数据和无标注数据，因此TrainDataset同时管理LabeledDataset和UnlabeledDataset两个数据集。

Semi-sklearn针对LabeledDataset和UnlabeledDataset分别设计了LabeledDataloader和UnlabeledDataloader两种数据加载器，而用一个TrainDataloader类同时管理两种加载器用于半监督学习的训练过程，除同时包含两个加载器外，还起到调节两个加载器之间关系的作用，如调节每一批数据中有标注数据与无标注数据的比例。

Semi-sklearn可以处理结构化数据、图像数据、文本数据、图数据四种现实应用中常见的数据类型，分别使用了四个与数据类型对应的组件StructuredDataMixin、VisionMixin、TextMixin、GraphMixin进行处理，对于一个数据集，可以继承与其数据类型对应的组件获得组件中的数据处理功能。

## 数据变换

使用机器学习算法利用数据学习模型和用模型对数据进行预测之前通常需要对数据进行预处理或数据增广，尤其是在半监督学习领域，部分算法本身就包含对数据进行不同程度的增广和加噪声的需求，Semi-sklearn的数据变换模块针对不同类型的数据提供了多样的数据预处理和数据增广方法，如对于结构化数据的归一化、标准化、最小最大化等，对于视觉数据的旋转、裁剪、翻转等，对于文本数据的分词、词嵌入、调整长度等，对于图数据的结点特征标准化、k近邻图构建、图扩散等。Semi-sklearn中所有数据变换方法都继承了sklearn中的TransformerMixin类，并且sklearn或pytorch都可以使用。对于依次进行的多次数据变换，sklearn的Pipeline机制和pytorch的Compose机制都可以使用。

## 算法使用

目前Semi-sklearn包含30种半监督学习算法，其中基于传统机器学习模型的算法13种（如图3-3所示）：半监督支持向量机类方法TSVM、LapSVM，基于图的方法Label Propagation、Label Spreading，生成式方法SSGMM，封装方法Self-Training、Co-Training、Tri-Training，集成方法SemiBoost、Assemble，半监督回归方法CoReg，半监督聚类方法Constrained K Means、Constrained Seed K Means；基于深度神经网络模型的算法17种（如图3-4所示）：一致性正则方法Ladder Network、Pi Model、Temporal Ensembling、Mean Teacher、VAT、UDA，基于伪标注的方法Pseudo Label、S4L，混合方法ICT、MixMatch、ReMixMatch、FixMatch、FlexMatch，生成式方法ImprovedGAN、SSVAE，图神经网络方法SDNE、GCN。

## 模型评估

Semi-sklearn提供了针对不同任务的不同评估指标，如针对分类任务的准确率、精度、召回率等，针对回归任务的均方误差、均方对数误差、平均绝对误差等，针对聚类任务的Davies Bouldin Index、Fowlkes and Mallows Index、Rand Index等。在Semi-sklearn中，评估方法可以在得到预测结果后调用，也可以用python字典的形式作为参数直接传入模型。

# 快速开始

## 数据加载
以CIFAR10数据集为例,首先CIFAR10类。

```python
from Semi_sklearn.Dataset.Vision.cifar10 import CIFAR10
```

实例化一个封装好的CIFAR10数据集,相当于一个数据管理器，root参数表示数据集存放地址，labeled_size参数表示有标注样本的数量或比例，stratified参数表示对数据集进行划分时是否要按类别比例划分，shuffle参数表示是否需要对数据集进行打乱，download参数表示是否需要下载数据集。

```python
dataset=CIFAR10(root='..\Semi_sklearn\Download\cifar-10-python',labeled_size=4000,stratified=False,shuffle=True,download=False)
```

通过init_dataset方法初始化数据集的内部结构，如果使用Semi-sklearn中的数据集，不需要设置参数，如果使用自定义数据集，需要传入具体的数据。

```python
dataset.init_dataset()
```

之后通过init_transform方法初始化数据预处理方式，这里直接采用默认设置。

```python
dataset.init_transform()
```

可以通过访问封装数据集参数的方法获取数据集中的具体数据。

```python
labeled_dataset=getattr(dataset,'labeled_dataset')
unlabeled_dataset=getattr(dataset,'unlabeled_dataset')
unlabeled_X=getattr(unlabeled_dataset,'X')
labeled_X=getattr(labeled_dataset,'X')
labeled_y=getattr(labeled_dataset,'y')
valid_X=getattr(dataset.valid_dataset,'X')
valid_y=getattr(dataset.valid_dataset,'y')
test_X=getattr(dataset.test_dataset,'X')
test_y=getattr(dataset.test_dataset,'y')
```

## 数据变换

以RandAugment数据增广为例，首先导入RandAugment类。

```python
from Semi_sklearn.Transform.RandAugment import RandAugment
```

对RandAugment进行实例化，参数n为进行随机增广的次数，表示增广的幅度，num_bins表示幅度划分的级别数。这里设置将增广幅度划分为10个等级，并采用第10级的增广增广2次。

```python
augmentation=RandAugment(n=2,m=10,num_bins=10)
```

之后输入数据完成数据增广。由两种方式：可以调用fit_transform()方法：

```python
augmented_X=augmentation.fit_transform(X)
```

也可以直接调用__call__()方法：

```python
augmented_X=augmentation(X)
```

## Pipeline机制

Semi-sklearn支持Pipeline机制，将多种数据处理方式以流水线的形式用于数据处理。
如在FixMatch算法中的强数据增广和弱数据增广。

```python
from sklearn.pipeline import Pipeline
from Semi_sklearn.Transform.RandomHorizontalFlip import RandomHorizontalFlip
from Semi_sklearn.Transform.RandomCrop import RandomCrop
from Semi_sklearn.Transform.RandAugment import RandAugment
from Semi_sklearn.Transform.Cutout import Cutout
weakly_augmentation=Pipeline([('RandomHorizontalFlip',RandomHorizontalFlip()),
                              ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
                              ])

strongly_augmentation=Pipeline([('RandAugment',RandAugment(n=2,m=5,num_bins=10,random=True)),
                              ('Cutout',Cutout(v=0.5,fill=(127,127,127))),
                              ('RandomHorizontalFlip',RandomHorizontalFlip()),
                              ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
                              ])
```

可以直接调用fit_transform()方法完成数据处理。

```python
weakly_augmented_X=weakly_augmentation.fit_transform(X)
strongly_augmented_X=strongly_augmentation.fit_transform(X)
```

## 训练一个基于经典机器学习的半监督模型

以Self-training算法为例。
首先导入并初始化BreastCancer数据集。

```python
from Semi_sklearn.Dataset.Table.BreastCancer import BreastCancer
dataset=BreastCancer(test_size=0.3,labeled_size=0.1,stratified=True,shuffle=True,random_state=0)
dataset.init_dataset()
dataset.init_transforms()
```

对数据进行预处理。

```python
labeled_X=dataset.pre_transform.fit_transform(dataset.labeled_X)
labeled_y=dataset.labeled_y
unlabeled_X=dataset.pre_transform.fit_transform(dataset.unlabeled_X)
unlabeled_y=dataset.unlabeled_y
test_X=dataset.pre_transform.fit_transform(dataset.test_X)
test_y=dataset.test_y
```

调用并初始化Self-training模型，以SVM模型为基学习器。

```python
from Semi_sklearn.Algorithm.Classifier.Self_training import Self_training
from sklearn.svm import SVC
SVM=SVC(C=1.0,kernel='linear',probability=True,gamma='auto')
model=Self_training(base_estimator=SVM,threshold=0.8,criterion="threshold",max_iter=100)
```

调用fit()方法进行模型训练。

```python
model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
```

对测试数据进行预测。

```python
result=model.predict(X=test_X)
```

对模型效果进行评估。

```python
from Semi_sklearn.Evaluation.Classification.Accuracy import Accuracy
from Semi_sklearn.Evaluation.Classification.Recall import Recall
print(Accuracy().scoring(test_y,result))
print(Recall().scoring(test_y,result))
```

## 训练一个基于深度学习的半监督模型

以FixMatch算法为例。
首先导入并初始化CIFAR10数据集。

```python
from Semi_sklearn.Dataset.Vision.cifar10 import CIFAR10
dataset=CIFAR10(root='..\Semi_sklearn\Download\cifar-10-python',labeled_size=4000,stratified=True,shuffle=True,download=False)
dataset.init_dataset()
dataset.init_transforms()
```

通过访问封装数据集参数的方法获取数据集中的具体数据。

```python
labeled_dataset=getattr(dataset,'labeled_dataset')
unlabeled_dataset=getattr(dataset,'unlabeled_dataset')
unlabeled_X=getattr(unlabeled_dataset,'X')
labeled_X=getattr(labeled_dataset,'X')
labeled_y=getattr(labeled_dataset,'y')
valid_X=getattr(dataset.valid_dataset,'X')
valid_y=getattr(dataset.valid_dataset,'y')
test_X=getattr(dataset.test_dataset,'X')
test_y=getattr(dataset.test_dataset,'y')
```

在深度学习中，需要使用数据加载器，首先需要将具体数据进行进行封装，并确定数据加载过程中的处理方式。

```python
from Semi_sklearn.Dataset.TrainDataset import TrainDataset
from Semi_sklearn.Dataset.UnlabeledDataset import UnlabeledDataset
train_dataset=TrainDataset(transforms=dataset.transforms,transform=dataset.transform,pre_transform=dataset.pre_transform,
                           target_transform=dataset.target_transform,unlabeled_transform=dataset.unlabeled_transform)

valid_dataset=UnlabeledDataset(transform=dataset.valid_transform)

test_dataset=UnlabeledDataset(transform=dataset.test_transform)
```

在初始化数据加载器之前，可以根据需求设置采样器,即sampler和batch_sampler，这里训练时采用随机采样，测试和验证时采用顺序采样。

```python
from Semi_sklearn.Sampler.RandomSampler import RandomSampler
from Semi_sklearn.Sampler.SequentialSampler import SequentialSampler
from Semi_sklearn.Sampler.BatchSampler import SemiBatchSampler
train_sampler=RandomSampler(replacement=True,num_samples=64*(2**20))
train_batch_sampler=SemiBatchSampler(batch_size=64,drop_last=True)
valid_sampler=SequentialSampler()
test_sampler=SequentialSampler()
```

以Pipeline形式设置数据增广方法，若存在多种增广方式，可以用python字典或列表存储。

```python
from sklearn.pipeline import Pipeline
from Semi_sklearn.Transform.RandomHorizontalFlip import RandomHorizontalFlip
from Semi_sklearn.Transform.RandomCrop import RandomCrop
from Semi_sklearn.Transform.RandAugment import RandAugment
from Semi_sklearn.Transform.Cutout import Cutout
weakly_augmentation=Pipeline([('RandomHorizontalFlip',RandomHorizontalFlip()),
                              ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
                              ])

strongly_augmentation=Pipeline([('RandAugment',RandAugment(n=2,m=5,num_bins=10,random=True)),
                              ('Cutout',Cutout(v=0.5,fill=(127,127,127))),
                              ('RandomHorizontalFlip',RandomHorizontalFlip()),
                              ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
                              ])
augmentation={
    'weakly_augmentation':weakly_augmentation,
    'strongly_augmentation':strongly_augmentation
}
```

之后设置深度学习中的神经网络结构，这里使用WideResNet作为神经网络的基本结构。

```python
from Semi_sklearn.Network.WideResNet import WideResNet
network=WideResNet(num_classes=10,depth=28,widen_factor=2,drop_rate=0)
```

设置深度学习中的优化器，这里使用SGD优化器。

```python
from Semi_sklearn.Opitimizer.SGD import SGD
optimizer=SGD(lr=0.03,momentum=0.9,nesterov=True)
```

```bash
echo "hello"
```

设置深度学习中的调度器用来在训练过程中调整学习率。

```python
from Semi_sklearn.Scheduler.CosineAnnealingLR import CosineAnnealingLR
scheduler=CosineAnnealingLR(eta_min=0,T_max=2**20)
```

在深度半监督学习算法中，可以用字典存储多个评估指标，直接在模型初始化时作为参数用于在训练过程中验证模型效果。

```python
from Semi_sklearn.Evaluation.Classification.Accuracy import Accuracy
from Semi_sklearn.Evaluation.Classification.Top_k_accuracy import Top_k_accurary
from Semi_sklearn.Evaluation.Classification.Precision import Precision
from Semi_sklearn.Evaluation.Classification.Recall import Recall
from Semi_sklearn.Evaluation.Classification.F1 import F1
from Semi_sklearn.Evaluation.Classification.AUC import AUC
from Semi_sklearn.Evaluation.Classification.Confusion_matrix import Confusion_matrix

evaluation={
    'accuracy':Accuracy(),
    'top_5_accuracy':Top_k_accurary(k=5),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_matrix(normalize='true')
}
```

初始化Fixmatch算法，并设置好各组件和参数。

```python
from Semi_sklearn.Algorithm.Classifier.Fixmatch import Fixmatch
model=Fixmatch(train_dataset=train_dataset,valid_dataset=valid_dataset,test_dataset=test_dataset,
               train_sampler=train_sampler,valid_sampler=valid_sampler,test_sampler=test_sampler,train_batch_sampler=train_batch_sampler,
               train_dataloader=train_dataloader,valid_dataloader=valid_dataloader,test_dataloader=test_dataloader,
               augmentation=augmentation,network=network,optimizer=optimizer,scheduler=scheduler,evaluation=evaluation,
               epoch=1,num_it_epoch=1,num_it_total=1,eval_it=2000,device='cpu',mu=7,
               T=1,weight_decay=0,threshold=0.95,lambda_u=1.0,ema_decay=0.999)
```

对模型进行训练，并在训练的同时对模型进行验证。

```python
model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X,valid_X=valid_X,valid_y=valid_y)
```

最后对测试数据进行预测。

```python
model.predict(test_X)
```

## 参数搜索

Semi-sklearn支持sklearn中的参数搜索机制。
首先初始化一个参数不完整的模型。

```python
model=Fixmatch(train_dataset=train_dataset,test_dataset=test_dataset,
               train_dataloader=train_dataloader,test_dataloader=test_dataloader,
               augmentation=augmentation,network=network,epoch=1,num_it_epoch=2,num_it_total=2,
               optimizer=optimizer,scheduler=scheduler,device='cpu',eval_it=1,
               mu=7,T=1,weight_decay=0,evaluation=evaluation,train_sampler=train_sampler,
               test_sampler=test_sampler,train_batch_sampler=train_batchsampler,ema_decay=0.999)
```

以字典的形式设置带搜索参数。
    
```python
param_dict = {"threshold": [0.7, 1],
              "lambda_u":[0.8, 1]
              }
```

以随机搜索的方式进行参数搜索。
首先进行搜索方式初始化。

```python
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_dict,n_iter=1, cv=4,scoring='accuracy')
```

开始参数搜索过程。

```python
random_search.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
```

## 分布式训练

可以采用分布式训练用多个GPU同时训练模型。以Fixmatch为例。
导入并初始化DataParallel模块。需要设置分布式训练所需的GPU。

```python
from Semi_sklearn.Distributed.DataParallel import DataParallel
parallel=DataParallel(device_ids=['cuda:0','cuda:1'],output_device='cuda:0')
```

初始化分布式训练的Fixmatch算法。

```python
model=Fixmatch(train_dataset=train_dataset,valid_dataset=valid_dataset,test_dataset=test_dataset,
               train_sampler=train_sampler,valid_sampler=valid_sampler,test_sampler=test_sampler,train_batch_sampler=train_batch_sampler,
               train_dataloader=train_dataloader,valid_dataloader=valid_dataloader,test_dataloader=test_dataloader,
               augmentation=augmentation,network=network,optimizer=optimizer,scheduler=scheduler,evaluation=evaluation,
               epoch=1,num_it_epoch=1,num_it_total=1,eval_it=2000,device='cpu',mu=7,parallel=parallel,
               T=1,weight_decay=0,threshold=0.95,lambda_u=1.0,ema_decay=0.999)
```

进行分布式训练。

```python
model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X,valid_X=valid_X,valid_y=valid_y)
```

## 保存和加载模型

可以使用pickle保存和加载半监督学习模型。
设置路径。

```python
path='../save/Fixmatch.pkl'
```

保存模型。

```python
with open(path, 'wb') as f:
    pickle.dump(model, f)
```

加载模型。

```python
with open(path, 'rb') as f:
    model = pickle.load(f)
```
# 用户指南

##  算法

### 基于经典机器学习的半监督学习算法

#### Generative Model

生成式半监督学习方法基于生成式模型，其假设数据由一潜在的分布生成而来，生成式方法建立了样本与生成式模型参数之间的关系，而半监督生成式方法将无标注数据的标注视为模型的隐变量数据，可以采用期望-最大化（EM）算法进行极大似然估计求解。


##### SSGMM

Shahshahani等提出了SSGMM模型。SSGMM即半监督高斯混合模型，假设数据由一个高斯混合模型生成，即样本的边缘分布可以表示为若干个高斯分布混合在一起的结果，且通过混合参数为每个高斯分布赋予一个权重。SSGMM将每一个类别的样本分布对应一个高斯混合成分。对于有标注数据，其类别对应的的高斯混合成分已知，对于无标注数据，其对应的高斯混合成分用一个概率分布表示，并可以将其分类为概率最高的高斯混合成分对应的类别。SSGMM假设样本服从独立同分布，其似然函数为所有有标注数据的样本与标注联合分布与所有无标注数据样本的边缘概率分布的乘积，通过最大似然估计使似然函数最大化，得到使当前有标注数据与无标注数据共同出现概率最大的生成式模型参数，包括高斯混合模型各部分的方差、均值以及权重。由于该方法存在无标注数据的标注这一无法观测的隐变量，因此无法直接求解最大似然参数，因此SSGMM采用了EM算法解决该问题。EM算法分为两步，其中E步根据当前参数与可观测数据得到未观测数据的条件分布或期望，在SSGMM模型中，这一步利用贝叶斯公式跟据已观测样本和当前模型的参数求解了无标注数据属于每一混合成分的概率，即无标注数据类别的条件分布；M步根据当前已观测变量的值与隐变量的期望或概率分布对模型参数进行最大似然估计，即原先由于隐变量未知无法进行最大似然估计，E步之后得到了隐变量的期望或条件概率分布，最大似然估计就变得可行了，在SSGMM模型中，这一步利用已观测到的有标注样本与标注、无标注样本与E步得到的类别条件分布更新了高斯混合模型的参数。E步与M步以一种迭代的形式交替进行，直至收敛，即可实现同时利用有标注数据与无标注数据训练一个高斯混合模型，并通过贝叶斯公式即可得到基于此高斯混合模型的分类器。

<!-- #### <font color=purple size=32>Semi-supervised Support Vactor Machine</font> -->
#### Semi-supervised Support Vactor Machine

支持向量机是机器学习领域最具代表性的算法之一。该类算法将二分类问题视为在样本空间中寻找合适的划分超平面。在线性可分的情况下，在所有能够完成正确分类的超平面中，最优的划分超平面应尽量位于不同类样本间的中间位置，可以提高模型对于未知样本进行预测的稳健性，即将各类样本中距离超平面最近的样本称为支持向量，不同类的支持向量距离超平面的距离相等，支持向量机算法的目的在于寻找距离其对应的支持向量最近的超平面。然而，在现实任务中，往往不存在一个可以将所有训练样本正确划分的超平面，即使存在，也难免会存在过拟合现象，因此一类支持向量机方法引入了软间隔（Soft Margin）机制，即允许超平面不必将所有样本正确分类，而是在优化目标中增加了对分类错误样本的惩罚。

半监督支持向量机是支持向量机算法在半监督学习理论的推广。半监督支持向量机引入了低密度假设，即学习得到的超平面除了需要基于有标注样本使分类尽可能分开，也要尽可能穿过所有样本分布的低密度区域，从而合理利用无标注样本。Semi-sklearn包含了两个半监督支持向量机方法：TSVM和LapSVM。

<!-- ##### <font color=blue size=16>TSVM</font> -->
#### TSVM

Joachims等[5]提出的TSVM是最基础的半监督支持向量机方法（如图2-1所示），是一种直推式方法。TSVM需要为每个无标注样本确定其标注，并寻找一个在有标注样本和无标注样本上间隔最大化的划分超平面。由于为无标注样本分配的标注并不一定是其真实有标注，在训练的初始阶段不能将无标注样本与有标注样本一视同仁，TSVM利用参数C_l和C_u分别代表对于有标注样本和无标注样本的惩罚量级，反应了对有标注样本和无标注样本重视程度。由于所有无标注样本的标注可能情况数量随无标注样本数量的增加呈指数级别上升，无法通过穷举的方式确定无标注样本的标注寻找全局最优解，TSVM使用了一种迭代的搜索方法为优化目标寻找近似解：首先基于有标注样本训练一个SVM，并用这个SVM对无标注样本进行预测；之后初始化C_l\llC_u，并开始迭代，在迭代过程中利用所有样本求解新的超平面，并不断寻找一对可能都发生错误预测的无标注异类样本交换标注并重新训练，直到不再能找到符合条件的异类样本，通过加倍C_u的值增加对无标注样本的重视程度，开始新一轮的迭代，直到C_u与C_l相等；最后将得到的模型对无标注样本的预测结果作为无标注样本的标注，完成直推过程。

<!-- ##### <font color=blue size=56>LapSVM</font> -->
#### LapSVM
Belkin等[6]基于<font color=red>**流形假设**</font>提出了LapSVM。经典的SVM算法追求使支持向量间隔最大化，这只考虑了样本在特征空间的分布情况，然而在实际应用中，高维空间中的样本往往都分布在低维的黎曼流形上，仅依赖于样本在特征空间的间隔进行划分的支持向量机容易忽视样本分布的本质特征。LapSVM在SVM的优化目标的基础上增加了一项流形正则化项，对样本的本质分布进行了引导。LapSVM在所有样本的基础上构建了图模型，通过样本的特征间的相似性得到图模型的权重矩阵，并计算其Laplace矩阵，通过Laplace正则项引导模型对于图中临近样本的预测结果尽可能一致。不同于TSVM，LapSVM仅对有标注样本的错误分类进行惩罚，但是在构建图模型时同时使用了所有样本，从而利用样本在流形上的分布使无标注样本参与了学习的过程。

#### Graph Based Method

基于图的半监督学习方法将数据集表示为一个图结构模型，以样本为节点，以样本间的关系为边，在半监督学习中，存在有标注数据与无标注数据，因此图中的结点有一部分存在标注，而另一部分没有标注，因此基于图的直推式半监督学习可以被视为标注在图中传播的过程。

##### Label Propagation

Zhu等[7]提出了Label Propagation算法。该算法以数据集中的样本为结点，样本间的关系为边进行全连接或k近邻构图，边权往往采用样本在原始数据空间或核空间（其中高斯核最为常用）的相似度进行表示。Label Propagation算法的目的在于将有标注数据的标注通过图结构向无标注数据传播，完成对无标注数据的预测，实现直推式半监督学习。Label Propagation的优化目标为模型预测结果在图结构中的Laplacian一致性正则项，即以边权为权重，将相邻节点间模型预测结果差异的加权均方误差作为优化目标。由于有标注数据的标注是固定的，因此Label Propogation仅需将Laplacian一致性正则项作为优化目标，求解无标注数据的标注使优化目标取最小值，即模型对于图上临近点的预测应该尽可能一致，使优化目标对无标注数据的标注求偏导数，使其偏导数为0，就可以得到其最优解，经证明这个通过直接计算得到的闭式最优解与不断迭代进行无限次标注传播最终收敛到的结果是一致的。通过直接的推导即可求得精确解，不需要模拟标注传递的过程，不需要为了收敛进行多次迭代，这也是Label Propagation对于其他图半监督学习方法的优势所在。

##### Label Spreading

Zhou等[9]提出了Label Spreading算法。不同于Label Propagation算法在传播过程中固定了有标注数据的标注，使其在整个传播过程中都保持不变，这保护了真实有标注数据对模型的影响，但是对于存在数据噪声的情况，Label Prapogation会存在一定的局限性，且Label Propagation算法中标注全部指向无标注数据，这可能会堵塞一些需要通过有标注数据结点进行传播的路径，使信息在图中的传播收到了一定的限制。而Label Spreading算法使标注可以在对所有临近结点进行广播，对于传播后结果与真实标注不符的有标注数据，Label Spreading会对其进行惩罚，而不是完全禁止。Label Spreading的优化目标有两项，第一项与Label Propagation的优化目标相同，但不存在模型对有标注数据预测结果必须等于其真实标注这一限制，第二项为对有标注数据预测损失的惩罚，需要设置一个惩罚参数作为其权重。由于优化目标不同，Label Propagation存在闭式解，而Label Spreading则需要通过迭代的方式求解，需要设置一个迭代折中参数，即每轮迭代都需要将本轮迭代得到的传播结果与样本的初始标注利用折中参数进行权衡作为当前预测结果。

#### Wrapper Method

不同于其他半监督学习方法用无标注数据与有标注数据共同学习一个学习器，封装方法基于一个或多个封装好的监督学习器，这些监督学习器往往只能处理有标注数据，因此封装类方法总是与无标注数据的伪标注密切相关。另外不同于其他方法需要固定学习器的形式，封装方法可以任意选择其监督学习器，具有非常强的灵活性，便于将已有监督学习方法扩展到半监督学习任务，具有较强的实用价值与较低的应用门槛。

##### Self-Training

Yarowsky等[8]提出了Self-Training方法。Self-Training方法是最经典封装方法，该方法是一种迭代方法，首先利用有标注数据训练一个监督分类器，然后再每一轮迭代中，用当前学习器对无标注数据进行预测，得到其伪标注，之后取自信度高于一定阈值的无标注样本及其伪标注与有标注数据结合，形成新的混合数据集，在混合数据集上训练新的分类器，用于在下一轮迭代过程中预测无标注数据的伪标注。在训练方法简单便捷，且可以使用任意可以提供软标注的监督学习器，为后续其他方法的研究提供了基础。


##### Co-Training

Blum等[10]提出了Co-Training。Co-Training即协同训练方法，用两个基学习器互相协同，辅助彼此的训练，即对于在一个学习器上自信度比较高的无标注样本，Co-Training会将该样本与其伪标注传递给另一个学习器，通过这种交互的形式，一个学习器上学到的知识被传递到了另一个学习器上。由于两个基学习器存在差异，其差异性决定了它们对于相同样本的学习难度会不同，Co-Training有效地利用了这种不同，使单独地学习器不仅可以使用其本身自信的伪标注，还可以使用另一个学习器自信的伪标注，加大了对无标注数据的利用程度，最后将两个学习器进行集成作为最终的预测结果。为了使两个基学习器有一定的差异，Co-Training采用了多视图假设，即基于相同数据的不同样本特征集合训练的模型应该对相同样本有相同的预测结果。Co-Training将样本的特征划分为两个集合，作为对样本从两个不同视图的观测，初始状态两个学习器的训练数据仅为不同视图下的有标注数据，在迭代过程中，在一个学习器上伪标注自信度高的无标注样本及其伪标注会被同时加入两个学习器的训练数据集，用于下一轮的训练，迭代持续至两个学习器的预测均不再发生变化为止。

##### Tri-Training

Zhou等[11]提出了Tri-Training方法。由于Co-Training等多学习器协同训练的方法必须要求基学习器之间需要存在差异，如数据视图不同或模型不同。但在实际应用中，可能只有单一视图的数据，且人为对原始数据进行特征切割会一定程度上损失特征间关系的相关信息，且对于划分方式需要一定的专家知识，错误划分可能会导致严重的性能下降；而采用不同类型的模型进行协同训练又需要设置多个监督学习器，考虑到封装方法较其他半监督学习方法的优势在于可以直接将监督学习算法扩展到半监督学习任务，因此实际应用中使用封装方法的场景往往只有一个监督学习算法，设计额外的监督学习算法一定程度上损失了封装方法的便捷性。Tri-Training方法从数据采样角度解决了这一问题，仅使用一个监督学习算法，对数据进行多次有放回随机采样（Boostrap Sample），生成多个不同的训练数据集，从而达到用同一算法学习得到多个不同模型的目的。不同于其他封装方法采用模型对无标注样本的自信度作为是否将无标注数据纳入训练数据集的依据，但是有些情况下模型会出现对错误分类过度自信的情况，导致伪标注与真实标注间可能存在较大的偏差，Tri-Training使用三个基学习器协同训练，在一个基学习器的训练过程中，对于无标注样本可以用另外两个基学习器判断是否应该将其纳入训练数据，如果在一轮迭代中，两个学习器的共同预测错误率较低且对于该无标注样本拥有相同的预测结果，那么这一无标注数据及其伪标注更可能会对当前训练的基学习器产生积极影响，使用两个模型的预测一致性判断是否使用无标注样本的方法相较仅使用一个模型的自信度的方法更加具备稳健性。另外不同于其他封装方法被选中的无标注数据会已知存在于训练数据中，这可能会导致被错误预测的无标注样本会对学习器造成持续性影响，永远无法被纠正，在Tri-Training算法中每一轮迭代采用的无标注数据集及其伪标注都会被重新选择。Tri-Training拥有扎实的理论基础，并基于理论基础在每轮迭代中对无标注数据的利用增加了限制条件，这些限制条件主要针对一个基学习器每一轮迭代训练中另外两个学习器的共同预测错误率和无标注数据的使用数量：当另外两个基学习器的共同预测错误率较高时，即使它们的预测结果具有一致性，都应该放弃在训练过程中使用无标注数据；当满足一致性的样本数量过多时，也应避免过度使用无标注数据，而是根据理论结果得到该轮迭代中使用无标注数据的数量上界，如果超出上界则应该通过采样进行缩减。对无标注数据使用的严格约束极大程度上增加了半监督模型的安全性，可以有效缓解因错误引入无标注数据导致模型性能下降的问题。

#### Ensemble Method

在机器学习领域，使用单个学习器容易因欠拟合或过拟合造成模型偏差或方差过高，使模型泛化能力不足，集成学习将多个弱学习器结合起来，既提高了模型对假设空间的表示能力，又减弱了因单一学习器的错误造成的影响，提高了模型的可靠性。在半监督学习领域，由于无标注数据的加入，使用单一学习器为无标注数据设置伪标注这一做法使单一学习器的不稳定性进一步加剧，对有效的集成学习方法有更强的依赖。

##### Assemble

Bennett等[12]提出了Assemble方法。Assemble即自适应监督集成，是基于自适应提升（AdaBoost）方法在半监督学习领域的扩展。提升（Boosting）方法是集成学习中的一类重要方法，这类方法通过当前集成学习器的预测效果对数据集进行采样，采样过程会更加重视目前集成学习器预测效果不佳的样本，用采样后的数据训练新一轮的学习器，这一策略使模型在每一轮新的弱学习器学习过程中可以更多关注目前集成学习器学习效果较差的样本，不断提高模型的泛化能力和稳健性。AdaBoost是Boosting类方法中最具代表性的方法，该方法根据模型预测结果与样本自身标注的差异自适应地调整样本权重，并根据样本权重对数据集进行采样用于学习下一轮迭代弱学习器的学习，并根据每一轮弱学习器的准确率确定其权重，加入到集成学习器中，其中准确率更高的弱学习器拥有更高的集成权重。ASSEMBLE通过对无标注数据添加伪标注的方法将AdaBoost方法在半监督学习领域进行了推广，初始阶段无标注样本的伪标注为与其最接近的有标注样本的标注，且无标注数据与有标注数据拥有不同的权重，在迭代过程中，每一轮无标注数据的伪标注被更新为该轮集成学习器的预测结果，随着迭代的进行，集成学习器效果越来越好，伪标注也越来越准确，进一步推动着新一轮弱学习器对集成学习器产生更有益的影响。

##### SemiBoost

Mallapragada等[13]提出了SemiBoost方法（如图2-3所示）。不同于Assemble方法仅将模型的预测结果与真实标注或伪标注之间的差异性作为对样本采样的依据，没有考虑样本之间的关系，SemiBoost基于图半监督学习方法，指出在采样时应该将样本间的相似度也纳入考量，应该对样本间相似度较高但目前集成学习器的预测结果不一致性较大的样本设置更大的权重，SemiBoost使集成学习器在不断迭代过程中，不仅提高了模型的泛化能力，还提高了模型对于相似样本预测的一致性，使模型更加稳健，这一过程中无标注样本发挥了更大的作用。SemiBooost每一轮迭代中对于新的弱学习器进行学习，其优化目标由两项组成，第一项以有标注样本集与无标注样本集间的相似度作为权重，惩罚了无标注数据的伪标注和与其相近的有标注数据的真实标注之间的差异性，这接近于标注传播的效果，使得模型可以根据图结构通过样本相似性和有标注数据的真实标注得到无标注数据的伪标注，使伪标注更大程度上接近真实标注；第二项以无标注样本集内部的相似度作为权重，对相似度较高的样本之间预测差异较大的无标注样本赋予更高的权重，缓解了噪声对模型的影响。

#### Semi-supervised Regression

目前大多数半监督学习算法都是针对分类任务而设计的，且不能自然地扩展到回归任务，仅有少部分工作针对半监督学习回归任务，这很大程度上是因为回归任务相较分类任务更难提出合理的假设，研究半监督回归相较半监督分类有着更多的困难。目前这一领域还有待更多的研究成果，半监督回归任务在现实场景中依然具备较大的需求和应用价值。

##### CoReg

Zhou等[14]提出了CoReg方法。CoReg将Co-Training算法引入了回归任务，在原本用于分类的封装类算法中，往往假设模型越自信的样本越会对之后的训练产生积极影响，因此将模型预测结果中类别概率的最大值作为模型对该样本的自信度，根据学习器对无标注数据预测的自信度选择一部分无标注样本及其伪标注加入到训练数据中参与之后迭代的训练，但是在回归任务中难以评估模型对无标注样本的自信度，因此难以选择无标注样本加入训练过程，这也是这一类方法难以应用于回归任务的一个重要原因。CoReg解决了这一问题，从而将Co-Training算法应用在了回归任务中。CoReg使用k近邻（kNN）模型作为基学习器，对于两个基学习器，为了保持它们之间存在差异性，分别使用了不同的阶数计算闵可夫斯基（Minkowsky）距离作为k近邻模型中样本间的距离。为了度量模型对样本的自信度，对于每一无标注样本，模型先预测其实值伪标注，将其与所有参与训练的样本结合起来重新训练一个学习器，并用均方误差损失评估该样本对它的k个近邻结点产生的影响，如果这些k近邻结点的均方误差降低，说明加入该无标注样本更可能对之后的训练产生更积极的影响，因此CoReg将加入一个无标注样本前后均方误差的差异作为自信度的评估标准，每一轮迭代将一个学习器上自信度最高的无标注样本及其实值伪标注加入到另一个学习器的训练数据中，从而完成了Co-Training的训练过程。

#### Semi-supervised Cluster

不同于半监督分类和半监督回归任务，用无标注数据辅助监督学习的过程，半监督聚类任务在原本无监督聚类的基础上引入了监督信息以辅助无监督学习的过程，其中监督信息不一定是有标注数据，也可能是其他与真实标注有关的知识，由于监督信息的不同也产生了不同多种半监督聚类方法。

##### Constrained k-means

Wagstaff等[15]提出了Constrained k-means算法。该算法在k-means聚类算法的基础上引入了称为必连（Must Link）和勿连（Connot Link）的约束作为监督信息，其中必连约束限制了一些样本必须属于同一聚类簇，而勿连约束限制了一些样本必须属于不同的聚类簇，且必连约束与勿连约束存在传递机制，如A与B必连且B与C必连则A与C必连，A与B勿连且B与C必连则A与C勿连。k-means算法将样本归属于簇中心与样本最近的簇，与之相似的是Constrained k-means算法也会优先考虑簇中心与样本最近的簇，但与之不同的是Constrained k-means算法会在将一样本归为一个簇时，会首先判断该样本与簇内样本间是否违反必连和勿连约束，如果违反，Constrained k-means会重新考虑下一个符合条件的簇，如果所有簇都不能满足约束，则会发出聚类失败的警告，需要随机选择不同的聚类中心重新初始化。

##### Constrained Seed k-means

Basu等[16]提出了Constrained Seed k-means算法。该算法不同于Constrained k-means将必连和勿连约束作为监督信息，而是直接采用了有标注数据作为监督信息。由于有了部分有标注数据，可以通过直接在有标注数据集上计算类别均值的方式计算聚类中心，这有效缓解了聚类算法中因初始聚类中心选择的随机性造成的聚类不稳定，且可以将有标注数据集上的类别数量作为聚类算法中的簇数k，不需要再人为选择k值，避免了聚类时不合理的簇数选择造成的聚类结果不理想。不同于k-means算法在迭代过程中对所有的样本根据其余目前所有簇中心的距离判断其应归属的簇，Constrained Seed k-means算法在迭代过程中仅对无标注数据所属的簇进行更新，对于有标注数据会根据其真实标注固定其所属的簇，不会因簇中心的变化而改变。使用有标注数据参于聚类过成时聚类器更加可靠，缓解了无监督聚类的盲目性，有效地减弱了聚类结果与样本真实标注间差距过大和由于随机性带来的不稳定现象。

### Deep Semi-supervised Learning

#### Consistency Regularization

深度学习方法通过设置损失函数，以梯度下降的优化方法引导模型训练的方向。一致性正则方法往往基于一致性假设，即假设对于样本增加一定程度的扰动，其预测结果应尽可能保持一致，从而在损失函数中引入了关于一致性的正则化项，使没有标注的样本也能参与到模型训练的过程中来，有助于提升模型对于噪声的稳健性。

##### Ladder Network

Rasmus等[17]提出了LadderNetwork方法（如图2-4所示）。该方法采用了自编码器结构，其中编码器最后一层的输出为分类软标注，即编码器同时具有分类功能，并采用两种编码方式，第一种为不带噪的编码器结构，即经典的编码器结构，第二种为带噪的编码器结构，即在经典的编码器基础上每一层的输入都会加入一定的噪声。LadderNetwork方法首先对样本分别进行带噪编码与不带噪编码，得到每个层次的带噪编码表示和不带噪编码表示；之后用解码器对带噪编码结果进行解码，得到每个层次的带噪解码表示；最后用均方误差损失（MSE）计算每一层次（包括原始输入数据作为第零层）的不带噪编码表示与带噪解码表示的不一致性，并通过原先确定的权重对各层次的不一致性进行加权作为无监督损失函数，从而利用无标注数据提升模型预测的稳健性。LadderNetwork算法的一致性正则化将带噪编码表示作为桥梁，惩罚了不带噪编码表示与带噪解码表示间的不一致性，一方面可以得到一个自编码器，使模型编码器与解码器各层次的表示可以保持一致，解码器利用编码后的结果可以尽可能地还原编码器各层表示以及原始数据；另一方面也可以使模型在存在噪声的情况下，保证隐层表示与没有噪声时尽可能一致，可以对抗微小的扰动。

##### UDA

Xie等[18]提出了UDA方法（如图2-5所示）。不同于LadderNetwork，UDA只对输入数据进行扰动，并不对隐层进行扰动，且UDA不一定采用高斯噪声进行扰动，而是可能可以采用多样的数据增广方式对数据进行增广。相比高斯噪声，UDA使用的数据增广，如图片旋转或文本替换等会对数据产生更大的影响，可以进一步提升模型的稳健性。UDA对无标注数据进行一次数据增广，之后比较增广前后的数据的预测结果，利用均方误差损失计算一致性正则项作为无监督损失，从而使无标注数据参与训练过程。

##### Pi Model

Laine等[19]提出了Pi Model方法（如图2-6所示）。不同于UDA将无标注数据进行一次增广后比较增广前后的数据的预测结果，计算一致性正则项，Pi Model分别对数据进行两次随机数据增广，并分别将两次增广的结果作为神经网络模型的输入进行预测，将预测结果的不一致性作为无监督损失，从而将无标注数据引入训练过程。由于增广过程的随机性，该方法两次增广会得到两项语义相似但特征可能略有不同的数据，通过一致性正则使模型对拥有一定界限的不同增广结果能产生相近的预测结果。

##### Temporal Ensembling

Laine等[20]还提出了Temporal Ensembling方法（如图2-7所示）。该方法对Pi Model进行了一些改进。在Pi Model中，对于每个无标注数据，Pi Model需要分别对其进行两次增广和两次伪标注预测以计算其结果的不一致性，这带来了较大的算力消耗。Temporal Ensembling方法将其中一次伪标注预测改为对历史伪标注的指数移动平滑（EMA），即对同一数据的历史预测结果进行加权平均从而将历史预测集成起来，其中每一轮的伪标注权重会随着后续轮次的增加以指数级别的速度衰减。这种集成方式在有效地保留了历史伪标注信息，通过计算当前伪标注与历史伪标注间的一致性作为得到函数，并在每一轮次结束时更新历史伪标注。EMA方法极大程度上保障了模型的稳健性，避免了模型过度受单轮预测的影响，也减慢了模型对历史信息的遗忘速度，且每一轮次中对于每个数据只需要进行一次增广和预测，历史信息仅需进行一次加权平均即可维护，相较Pi Model极大地减少了算力消耗。

##### Mean Teacher

Tarvainen等[21]提出了Mean Teacher方法（如图2-7所示）。该方法借助了知识蒸馏的思想，即将教师模型的预测结果作为伪标注，用于训练学生模型，确保教师模型与学生模型预测结果的一致性，从而将知识由较为复杂的教师模型蒸馏到较为简单的学生模型。经典的知识蒸馏方法的目的在于模型的简化，即教师模型采用较为复杂的模型，学生模型采用较为简单的模型，而Mean Teacher的目的在于通过一致性使无标注数据参与学习过程并提升模型的稳健性，因此教师模型并非是复杂模型，而是在学生模型的基础上对参数进行指数移动平滑，这相对于经典的知识蒸馏方法减少了计算开销。Temporal Ensembling方法对每一轮次的预测结果进行指数移动平滑的计算，但是只有在每一轮次结束时才会对整体的历史信息进行更新，对于大型的数据集，会导致历史信息不能及时对同一训练轮次（Epoch）后续批次（Batch）的数据产生影响，会导致对历史信息的利用不够及时的问题。不同于Temporal Ensembling，Mean Teacher改为对模型参数采用指数平滑计算，在每一批次训练结束后都会及时更新模型参数的历史信息，有效地解决了历史信息更新与利用不及时的问题，这使得Mean Teacher方法更加灵活，通用性更强。

##### VAT

Miyato等[22]提出了VAT。不同于对数据增加随机噪声的方法，VAT的思想在于对数据增加对抗噪声，使模型在数据受一定限制条件下噪声影响时的最坏表现可以更好，这对应了博弈问题中的零和博弈问题和优化问题中的最小最大化问题。对于经典的监督对抗算法，通常将真实标注与模型预测结果之间的交叉熵损失作为对抗优化的目标，首先通过内层优化得到对于当前模型和数据使损失最大的噪声，之后通过外层优化得到在对数据施加噪声的情况下的模型参数，内外优化交替进行，使模型在应对数据噪声时可以在最坏情况下表现得不会太差。其中，外层优化为对模型参数得优化，往往通过梯度下降来进行，而内部优化是针对数据噪声的优化，该优化不存在闭式解，且因针对不同数据应采用不同的对抗噪声，不适宜用梯度优化，需要对最优噪声进行近似，在经典的监督对抗算法中常采用线性近似，即先对无噪声数据进行预测并计算损失函数的值，进行梯度回传，得到对于无噪声数据的梯度，并将归一化后的梯度与噪声上界的乘积最为对抗噪声。
不同于经典的监督对抗算法，VAT需要解决半监督场景存在无标注数据的问题，即无法通过监督计算损失后回传梯度计算对抗噪声，为了解决这一问题，VAT算法采用了一致性策略，即将监督损失改为一致性损失，将利用真实标注计算损失改为利用模型分别对无噪声数据和噪声数据进行预测得到噪声伪标注与无噪声伪标注，计算二者间的一致性作为无监督损失。在VAT算法中，不同于监督的对抗算法，对于无标注数据内层优化无法采用线性近似，这是因为在监督的对抗算法的内层优化中，首先需要计算真实标注与模型对不加噪数据预测结果间的分类损失，而VAT用伪标注代替了真实标注，导致对于不加噪数据回传的梯度始终为0，无法得到梯度方向从而无法得到对抗噪声，因此VAT采用了二阶泰勒近似代替了线性近似，将计算对抗噪声的问题转化为了计算损失函数对于噪声的海森矩阵的主特征向量的问题。由于对于d维数据噪声，计算其海森（Hessian）矩阵的特征向量需要O\left(d^3\right)的时间复杂度，为了解决计算复杂度过高的问题，VAT采用了幂迭代（Power Iteration）方法求解近似的矩阵特征向量，即先对近似特征向量进行随机采样，并不断用矩阵与当前近似特征向量相乘得到新的近似特征向量，不断进行该过程即可在较低时间消耗的情况下得到较好的近似结果，为了进一步避免对海森矩阵的直接计算，VAT采用了有限差分（Finite Difference）方法近似求解矩阵与近似特征向量的乘积。相较其他基于一致性正则的方法，VAT方法采用对抗噪声比采用随机噪声可以进一步提升模型的稳健性，避免了随机性对实验结果的过度干扰，基于对抗的方法更加可靠地保证了模型在最坏情况下的表现，具有更好的理论基础，且VAT在计算对抗噪声时通过近似方法避免了过度的额外计算开销，并解决了监督对抗算法无法直接应用于无标注数据的困境。

#### Pseudo Labeling

基于伪标注的方法通过为无标注数据赋以伪标注从而使无标注数据对学习过程产生影响。且由于模型对于不同样本的自信度不同，基于伪标注的方法通常取自信度较高的样本与其伪标注参与训练过程。

##### Pseudo Label

Lee等[23]提出了Pseudo Label方法（如图2-10所示）。该方法为最基础的伪标注方法，其损失函数包括两项，分别是监督损失和无监督损失，两部分都是交叉熵损失函数。其中对于无标注数据，Pseudo Label方法对神经网络输出结果进行softmax运算，得到样本属于各类别的自信度，Pseudo Label取自信度最高的类别作为样本的伪标注，用伪标注计算交叉熵损失。另外，在每一轮次中，并不是所有无标注样本都会参与训练过程，Pseudo Label设置了一个阈值，只有当本轮无标注样本的伪标注自信度大于所设阈值时，才会参加训练过程。Pseudo Label还设置了超参数用于控制监督损失与无监督损失的比重，并采用了预热（warmup）机制，刚开始训练时，无监督损失比重较低，随着训练的进行，无监督损失的比重越来越大。

##### S4L

Beyer等[24]提出了S4L方法（如图2-11所示）。这一方法采用了自监督技术，其基本思想在于：无标注数据无法直接参与分类器的训练，但是可以利用自监督对表示层产生影响，使模型可以学到更好的隐层表示，从而有助于分类器的训练。该方法主要用于图像数据，随机取0^\circle、90○、180○、270○之一作为度数对图像进行旋转操作，将角度作为伪标注，旋转后的图像与角度即可形成数据对，可以训练一个对角度进行分类的神经网络模型，虽然神经网络最后的分类层与目标任务不同，但其学到的隐层表示有助于对真实任务的学习。对于有标注数据，S4L也会进行同样的处理，使其拥有两个标注，代表旋转度数的伪标注和用于目标任务的真实标注，S4L对有标注数据用了两个分类层，其中之一是与无标注数据共享的度数分类器，另一个是用于目标任务的真实类别分类器，两个分类层共用相同的隐层。通过上述方式，S4L在训练自监督分类器的同时，使模型可以学习到更好的表示，从而提升了模型对目标任务的分类效果。与预训练与微调范式不同，S4L不用提前训练模型，而是可以同时处理有标注数据和无标注数据，并且相互促进，且有标注数据也参与了自监督学习过程，对数据进行了更大程度的利用。S4L也可以推广到其他类型的数据，需要采用与之对应的自监督训练方法。

#### Hybird Method

不同的半监督学习技术之间往往不会冲突，很多常用的半监督学习算法不局限于仅使用一类技术，而是将一致性、伪标注等技术进行结合，各取所长，产生新的混合方法。混合方法可以同时利用不同技术的优势，从而达到更好的训练效果。由于同时使用了多种技术，混合方法往往更加具备通用性。

##### ICT

Verma等[25]提出了ICT方法（如图2-12所示）。ICT即插值一致性训练，通过Mixup[34]数据增广方法对数据与预测结果进行线性插值，通过比较模型对插值后样本的预测结果与模型对原始数据的预测结果的插值之间的一致性将无标注数据引入训练过程。Mixup由Beta分布生成一个混合参数，对两项数据按这一混合参数得到线性插值，得到两项数据的混合数据，以此实现数据增广。ICT方法的损失函数分为监督损失与无监督损失两部分，其中监督损失通过交叉熵函数计算，无监督损失则要通过插值一致性计算。对于每一批次的数据，ICT首先根据Beta分布采样一个混合参数，然后将该批次样本随机打乱，将打乱的批数据与未打乱的批数据以混合参数为比例进行Mixup混合，得到混合批数据，模型对未打乱批数据和混合批数据进行预测，得到未打乱预测结果与混合预测结果，并将未打乱预测结果按样本打乱顺序重新排列得到打乱预测结果，ICT将未打乱预测结果与打乱预测结果以和样本相同的混合参数进行线性插值，并将插值结果与混合预测结果间的不一致性作为无监督损失。对于混合后的无标注数据，ICT使模型输出的软标注接近于伪标注的混合，将一致性技术与伪标注技术结合起来，使模型更加稳健。

##### MixMatch

Berthelot等[26]提出了MixMatch方法（如图2-13所示）。该方法也用了Mixup方法，但不同于ICT仅对无标注数据的样本与伪标注进行Mixup，MixMatch对有标注数据与无标注数据进行了混合，并对混合后的数据样本及其标注与伪标注进行了Mixup。MixMatch首先对无标注数据多次增广并进行多次预测，通过对多次预测结果求均值并进行锐化得到无标注数据的伪标注，对数据进行多次不同增广使模型的伪标注更加具备可靠性，对伪标注进行锐化降低了标注分布的熵，使分类界限尽可能穿过样本的低密度区域；之后MixMatch对有标注数据与无标注数据进行了结合与打乱，使无标注数据集与有标注数据集形成了一个新的混合数据集，从混合数据集中取出与原有标注数据集相同数量的数据进行Mixup作为新的有标注数据集，将混合数据中剩余数据与无标注数据集进行Mixup得到新的无标注数据集；最后MixMatch分别对新有标注数据集和新无标注数据集进行预测，用新有标注数据集的预测结果计算交叉熵作为监督损失，用新无标注数据的预测结果计算均方误差作为无监督损失，通过权重参数将二者结合起来作为模型的损失函数。不同于其他方法将有标注数据与无标注数据分别计算损失，MixMatch将有标注数据与无标注进行了结合、打乱、重新划分，这降低了因错误的伪标注导致模型性能下降的风险。在原本仅使用伪标注训练的过程中加入真实标注，有助于利用真实标注辅助无标注数据的训练，引导无标注一致性的正确训练方向，既保障了一致性正则原有的稳健性，还能使模型不会因伪标注与真实标注不符过度偏离目标。

##### ReMixMatch

Berthelot等[27]还提出了ReMixMatch方法（如图2-14所示）。ReMixMatch是MixMatch的改进版本，其引入了两种技术：分布对齐和增广锚定。分布对齐目的在于使模型对于无标注数据预测得到的伪标注应与有标注数据的标注有相同的边缘概率分布，在深度学习中，模型的预测经常偏向数量较多的类别，另外MixMatch对软标注使用了锐化操作减少了标注分布的熵以促使分类边界尽可能通过低密度区域，这都导致了有标注数据的标注分布与无标注数据的伪标注分布产生了差异，这反映了为无标注数据赋予伪标注存在类别间的不公平现象，分布对齐技术有效缓解了这样的问题。分布对齐技术计算有标注数据的真实标注分布，在每一批次的训练中，计算其输出的软标注分布，对于一个样本的软标注，使其与真实标注分布与当前批次软标注分布的比值相乘得到对齐后的软标注，将对齐后的软标注进行锐化得到样本的伪标注。增广锚定是为了使模型适应更强的数据增广，对于监督学习方法，在一定程度内，对数据施加更强的数据增广可以进一步提升模型的泛化能力，但这是以监督学习中无论对样本施加强增广还是弱增广，标注都不会发生变化为前提。在半监督学习中，往往由模型对无标注数据的预测结果得到伪标注，伪标注会随着数据增广的形式而变化，如果对样本施加较强的增广，容易使伪标注过度偏离真实标注，无法发挥监督学习中强数据增广的作用，这也导致了MixMatch方法不能与较强的数据增广方式相容，ReMixMatch通过引入增广锚定技术首先对无标注样本进行弱数据增广，将模型对其预测的结果作为伪标注，并将其作为“锚”固定下来，这使得后续无论对无标注数据进行何种数据增广，都不会使其伪标注发生变化。ReMixMatch方法对无标注数据进行了一次弱数据增广和多次强数据增广，并都以模型对弱增广数据的预测结果经对齐与锐化后作为伪标注，由弱增广和所有强增广后的数据集组成更大的无标注数据集。之后ReMixMatch采用与MixMatch相同的策略对有标注数据集和无标注数据集进行组合、打乱与重新划分。另外，ReMixMatch的损失函数与MixMatch由较大的不同，ReMixMatch的有监督损失与无监督损失均采用交叉熵进行计算，且不同于MixMatch的损失函数仅包含监督损失与无监督损失两项，ReMixMatch增加了两项损失，这是由于MixMatch仅对Mixup后的数据集进行损失计算，虽然Mixup使模型拥有了更好的泛化性能，但是仅使用Mixup后的数据可能会忽略Mixup前数据集的一些信息，因此ReMixMatch从多个Mixup前的强增广数据集中取出一个，用于计算Mixup前数据的无监督损失作为损失函数第三项；ReMixMatch还借鉴了S4L的自监督策略，对取出的Mixup前的强增广数据集进行随机旋转并对其旋转角度进行预测，自监督进一步促进了模型隐层的学习，将对旋转角度分类的交叉熵损失作为自监督损失，用作损失函数的第四项。ReMixMatch以一个更为复杂的框架将多种技术融为一体，不仅结合了各方法的优势，且因为其全面性而更加通用。

##### FixMatch

Sohn等[28]提出了FixMatch方法（如图2-15所示）。FixMatch同样使用了强数据增广与弱数据增广，不同于ReMixMatch通过增广锚定技术利用弱数据增广固定无标注数据的伪标注，FixMatch更加关注模型对弱增广数据与强增广数据预测结果的一致性。与ReMixMatch相同的是FixMatch同样根据模型对弱增广数据的预测结果得到伪标注，FixMatch的伪标注为硬标注。之后FixMatch对无标注数据进行强增广，得到预测结果，FixMatch仅用模型自信的无标注数据进行训练，即设置一个阈值参数，仅当自信度大于阈值参数时，该数据才会参与训练过程。FixMatch利用模型对弱增广样本得到的伪标注和模型对强增广样本得到的预测结果计算交叉熵作为无监督损失，通过权重参数将无监督损失与监督损失结合起来作为FixMatch的损失函数。

##### FlexMatch

Zhang等[29]提出了FlexMatch方法（如图2-16所示）。FlexMatch是对于FixMatch的改进，且注重于解决半监督学习中各类别间的不公平现象，FixMatch根据固定的阈值参数筛选自信度高的无标注样本及其伪标注参与模型训练过程，但有时虽然原始数据集是类别平衡的，但由于各类别学习难度不同，采用固定阈值进行筛选会导致一些难学习的类别相较易学习的类别更少参与训练过程，这样模型对较难学习的类别样本自信度更低，进一步加剧了参与训练的无标注数据的类别不平衡，这种不公平形成了恶性循环，造成了马太效应，导致模型对较难学习的类别学习效果越来越差，因此FlexMatch提出了对于不同的类别应采用不同的筛选标准，缓解因学习难度不同造成的类别不平衡现象。FlexMatch在FixMatch的基础上改用了动态阈值设置的方法，对较难学习的类别设置更低的自信度阈值，一种最基础的方法为设置一个验证集，根据模型在验证集上各类别的准确率设置阈值，但由于有标注的训练数据本身已较少且在训练过程中不断进行验证更新模型的验证准确率会造成较大的计算开销，因此FlexMatch采用了近似评估类别准确率的方法，首先选取自信度最高的类别作为其样本伪标注，对于每一批次的无标注数据，统计不同类别在该批数据中作为伪标注且自信度大于阈值参数的数量，之后对不同类别的统计数量除以其中的最大值进行归一化作为该类别的分类难度的评估度量，用固定阈值与该类别的分类难度度量相乘即可得到该类别在这一批次无标注数据中应使用的动态阈值。FlexMatch较好地缓解了无标注数据根据自信度进行筛选后由于学习难度不同造成的的类别不平衡问题，且没有因在训练过程中评估模型对不同类别的预测效果产生过多的额外计算时间和存储开销。

#### Deep Generative Model

生成式方法利用真实数据对数据分布进行建模，并且可以利用这一分布生成新的数据。不同于经典的生成模型，深度生成式模型基于深度神经网络生成数据。生成对抗网络（GAN）和变分自编码器（VAE）是最常用的生成式模型。

##### ImprovedGAN

生成对抗网络模型分为两个部分：生成器和判别器，其中生成器假设数据可以由产生于某一特定的分布的低维隐变量生成，通过从隐变量分布上随机采样用于生成模拟数据，而生成器是一个分类器，用于判别输入样本是真实数据还是由生成器生成的模拟数据，生成器通过优化要使生成的样本与真实样尽可能接近以欺骗判别器，判别器通过优化要尽可能正确地区分真假样本，避免被生成器欺骗，两者以对抗地方式共同训练，从而达到同时得到较好的生成器与判别器的目的。

Salimans等提出了ImprovedGAN （如图2-17所示）。经典的GAN模型仅利用无标注数据就可以完成训练，其判别器仅需要判断样本是真实样本还是生成样本。ImprovedGAN加入了对有标注数据的利用，要求判别器不仅要区分样本的真实性，还要完成对样本的分类，即将判别器改为k+1类分类器，其中k是原数据集的类别数量，通过生成器与判别器的交替训练，既可以实现数据生成，又可以完成分类。

##### SSVAE

变分自编码器将深度自编码器融入生成模型，同样假设存在产生于某一特定分布的低维隐变量，将隐变量作为原特征的表示向量，并通过深度神经网络建立隐变量到原特征的映射，作为解码器；由于无法直接求得原始特征到隐变量的后验概率，也需要通过神经网络来近似，作为编码器，学习的目标为做大化原始样本的边缘概率，由于当近似后验分布与真实后验分布相等时， 边缘概率可以达到其上界，因此可以学到与真实后验分布近似的祭祀后验分布，作为编码器可以得到合理的样本表示。

Kingma等提出了SSVAE。经典的VAE模型仅利用无标注数据就可以完成训练，其目标在于通过编码器完成对数据表示的学习，并通过解码器可以实现数据生成。SSVAE加入了对有标注样本的应用，将编码器分为了两个部分，第一部分对原始数据进行编码得到样本软标注的概率分布，第二部分将原始数据与软标注共同作为输入得到隐变量的概率分布。经典VAE模型的编码器仅对数据的表示进行学习，SSVAE的编码器首先可以用样本进行分类，之后可以结合样本信息与类别信息共同学习样本的表示。


#### Deep Graph Based Method

对于原始数据是图数据的情况，由于实例之间并非独立关系，而是通过边相连，经典的深度学习方法无法有效利用图模型的结构信息，因此无法直接将其应用于图数据。然而，图数据在实际应用中非常常见，研究可以用于图数据的深度学习方法具有重要意义，目前图深度学习已经取得了一定的研究成果。在半监督领域也是如此，经典的半监督学习方法忽略了图的结构信息，因此直接用于图结构数据效果并不理想，现实中的图数据任务往往都是半监督的，即待预测结点与训练节点在一张图上，图中同时存在有标注数据与无标注数据。

##### SDNE

Wang等[32]提出了SDNE （如图2-18所示）。SDNE是一种可以在图中结点没有特征表示，仅有图结构信息的情况下学习图中结点嵌入向量的半监督图深度学习方法。该方法采用了自编码器结构，取结点在邻接矩阵中对应的行作为结点的邻接向量，将结点的邻接向量作为结点的特征输入自编码器，通过编码器得到结点的嵌入表示，通过解码器还原邻接向量，对于整个图，相当于通过自编码器还原了邻接矩阵。SDNE的损失函数主要包含三项：第一项惩罚了自编码器输入与输出的不一致性，使邻自编码器的输入与输出尽可能一致，另外与经典自编码器不同的是，SDNE的输入是邻接向量，由于邻接矩阵的稀疏性，导致输入的特征中存在大量的零值，SDNE指出应该更加关注对于非零值的还原，因此赋予了零值与非零值不同的权重；第二项为拉普拉斯正则，根据图结构信息惩罚了相邻节点间隐层表示的不一致性，并将邻接矩阵作为权重，得到了拉普拉斯正则项；第三项为L2正则，惩罚了自编码器的参数复杂度，以此来避免过拟合。在SDNE方法中，损失函数的第一项更关注结点本身的特征，而第二项更关注相邻节点间的信息，即图的结构信息，有效解决了经典半监督学习算法无法有效利用图结构信息的问题。


##### GCN

Kipf等[33]提出了GCN。与SDNE使用结点的邻接向量作为结点特征学习嵌入表示不同，GCN更适用于结点本身存在特征的情况，GCN可以同时利用结点本身的特征信息和图结构信息进行学习，显著地提升了模型的效果。在图深度学习中，图神经网络（GNN）[35]是最常用的一类方法，这类方法通常以存在结点特征的图作为输入，可以学习到结点的深层表示，并以此完成学习任务。经典的GNN方法分为两个步骤：第一个步骤为聚集（Aggregate），即通过图结构将近邻结点的信息进行汇集；第二个步骤为更新（Update），即根据结点自身表示与近邻结点更新结点表示。不断重复这两个步骤，可以得到每个结点的深层表示，由于聚集操作存在传播效果，结点的深层表示中不仅涵盖了节点自身信息，还涵盖了图结构信息。经典的聚集操作为线性聚集，即将近邻节点表示的线性组合作为该节点的近邻表示，经典的更新操作为使用感知机模型，由结点自身表示与近邻表示得到新的自身表示。经典的GNN模型存在一定的局限性，其对近邻节点的表示进行线性组合的聚集方式使度较大的结点更大程度地影响了其他节点，而度较小的结点对整个训练过程的影响较小。GCN方法对每一结点将标准化后的近邻表示与自身表示直接相加，并将结果作感知器的输入，得到的结果作为新的自身表示，其中标准化过程将近邻结点与自身结点的表示分别除以一个标准化因子，其中近邻结点的标准化因子为自身结点的度与近邻结点的度的几何平均，自身结点的标准化因子为自身结点的度。GCN在图结构任务上有着优异的表现，并且其更新过程避免了对近邻结点线性组合权重的学习，拥有更少的参数与更高的效率。

# 参考文献

[1]	VAN ENGELEN J E, HOOS H H. A survey on semi-supervised learning[J]. Machine Learning, 2020, 109(2): 373-440.

[2]	OUALI Y, HUDELOT C, TAMI M. An Overview of Deep Semi-Supervised Learning[J/OL]. arXiv:2006.05278 [cs, stat], 2020[2022-03-01]. http://arxiv.org/abs/2006.05278.

[3]	YANG X, SONG Z, KING I, et al. A Survey on Deep Semi-supervised Learning[J/OL]. arXiv:2103.00550 [cs], 2021[2022-03-01]. http://arxiv.org/abs/2103.00550.

[4]	SHAHSHAHANI B M, LANDGREBE D A. The Effect of Unlabeled Samples in Reducing the Small Sample Size Problem and Mitigating the Hughes Phenomenon[J]. IEEE Transactions on Geoscience and remote sensing, 1994, 32(5): 1087-1095.

[5]	JOACHIMS T. Transductive Inference for Text Classi cation using Support Vector Machines[C].  International Conference on Machine Learning, 1999, 99.

[6]	BELKIN M, NIYOGI P, SINDHWANI V. Manifold Regularization: A Geometric Framework for Learning from Labeled and Unlabeled Examples[J]. Journal of machine learning research, 2006, 7(11).

[7]	ZHU X, GHAHRAMANI Z. Learning from Labeled and Unlabeled Data with Label Propagation[R], 2002.

[8]	ZHOU D, BOUSQUET O, LAL T, et al. Learning with Local and Global Consistency[C]. Advances in Neural Information Processing Systems, 2003, Vol. 16.

[9]	YAROWSKY D. Unsupervised Word Sense Disambiguation Rivaling Supervised Methods[C]. 33rd Annual Meeting of the Association for Computational Linguistics. Cambridge, Massachusetts, USA: Association for Computational Linguistics, 1995: 189-196.

[10]	BLUM A, MITCHELL T. Combining labeled and unlabeled data with co-training[C]. Proceedings of the eleventh annual conference on Computational learning theory. Conference on Learning Theory, 1998: 92-100.

[11]	ZHI-HUA ZHOU, MING LI. Tri-training: exploiting unlabeled data using three classifiers[J]. IEEE Transactions on Knowledge and Data Engineering, 2005, 17(11): 1529-1541. 

[12]	BENNETT K P, DEMIRIZ A, MACLIN R. Exploiting Unlabeled Data in Ensemble Methods[C]. Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, 2002.

[13]	MALLAPRAGADA P K, RONG JIN, JAIN A K, et al. SemiBoost: Boosting for Semi-Supervised Learning[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2009, 31(11): 2000-2014.

[14]	ZHOU Z H, LI M. Semi-Supervised Regression with Co-Training[C]. International Joint Conference on Artificial Intelligence, 2005, 5.

[15]	WAGSTAFF K, CARDIE C, ROGERS S, et al. Constrained K-means Clustering with Background Knowledge[C]. International Conference on Machine Learning, 2001, 1.

[16]	BASU S, BANERJEE A, MOONEY R. Semi-supervised Clustering by Seeding[C]//In Proceedings of 19th International Conference on Machine Learning. 2002.

[17]	RASMUS A, BERGLUND M, HONKALA M, et al. Semi-supervised Learning with Ladder Networks[C]. Advances in Neural Information Processing Systems, 2015, 28.

[18]	XIE Q, DAI Z, HOVY E, et al. Unsupervised Data Augmentation for Consistency Training[C]. Advances in Neural Information Processing Systems, 2020, 33: 6256-6268.

[19]	LAINE S, AILA T. Temporal Ensembling for Semi-Supervised Learning[C]. International Conference on Learning Representations, 2017, 4(5): 6.

[20]	TARVAINEN A, VALPOLA H. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results[C]. Advances in Neural Information Processing Systems, 2017, 30.

[21]	MIYATO T, MAEDA S ichi, KOYAMA M, et al. Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning[J]. IEEE transactions on pattern analysis and machine intelligence, 2018, 41(8): 1979-1993.

[22]	LEE D H. Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks[C]. ICML 2013 Workshop : Challenges in Representation Learning (WREPL), 2013, 3(2): 869.

[23]	ZHAI X, OLIVER A, KOLESNIKOV A, et al. S4L: Self-Supervised Semi-Supervised Learning[C]. Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 1476-1485.

[24]	VERMA V, KAWAGUCHI K, LAMB A, et al. Interpolation Consistency Training for Semi-Supervised Learning[C]. International Joint Conference on Artificial Intelligence, 2019: 3635-3641

[25]	BERTHELOT D, CARLINI N, GOODFELLOW I, et al. MixMatch: A Holistic Approach to Semi-Supervised Learning[C]. Advances in Neural Information Processing Systems, 2019, 32.

[26]	ZHANG B, WANG Y, HOU W, et al. Flexmatch: Boosting semi-supervised learning with curriculum pseudo labeling[J]. Advances in Neural Information Processing Systems, 2021, 34.

[27]	SOHN K, BERTHELOT D, LI C L, et al. FixMatch: Simplifying Semi-Supervised Learning with Consistency and Conﬁdence[J]. 21.

[28]	BERTHELOT D, CARLINI N, CUBUK E D, et al. ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring[J/OL]. arXiv:1911.09785 [cs, stat], 2020[2022-03-02]. http://arxiv.org/abs/1911.09785.

[29]	SALIMANS T, GOODFELLOW I, ZAREMBA W, et al. Improved Techniques for Training GANs[C]. Advances in Neural Information Processing Systems, 2016, 29.

[30]	KINGMA D P, REZENDE D J, MOHAMED S, et al. Semi-Supervised Learning with Deep Generative Models[C]. Advances in neural information processing systems, 2014, 27.

[31]	WANG D, CUI P, ZHU W. Structural Deep Network Embedding[C]. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016: 1225-1234.

[32]	KIPF T N, WELLING M. Semi-Supervised Classification with Graph Convolutional Networks[C]. International Conference on Learning Representations, 2017.

[33]	PEDREGOSA F, VAROQUAUX G, GRAMFORT A, et al. Scikit-learn: Machine Learning in Python[J]. The Journal of Machine Learning Research, 2001, 12: 2825-2830.

[34]	ZHANG H, CISSE M, DAUPHIN Y N, et al. mixup: Beyond Empirical Risk Minimization[C]. International Conference on Learning Representations, 2018. 

[35]	SCARSELLI F, GORI M, TSOI A C, et al. The graph neural network model[J]. IEEE transactions on neural networks, 2008, 20(1): 61-80.

[36]	GASTEIGER J, WEISSENBERGER S, GÜNNEMANN S. Diffusion Improves Graph Learning[J/OL]. arXiv:1911.05485 [cs, stat], 2022. http://arxiv.org/abs/1911.05485.

[37]	DAVIES D, BOULDIN D. A Cluster Separation Measure[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1979, 2: 224-227. 

[38]	FOWLKES E B, MALLOWS C L. A Method for Comparing Two Hierarchical Clusterings[J]. Journal of the American Statistical Association, 1983, 78(383): 553-569. 

[39]	RAND W M. Objective Criteria for the Evaluation of Clustering Methods[J]. Journal of the American Statistical Association, 2012, 66(336): 846-850.

[40]	ZAGORUYKO S, KOMODAKIS N. Wide Residual Networks[J/OL]. arXiv:1605.07146 [cs], 2017[2022-04-26]. http://arxiv.org/abs/1605.07146.

[41]	CUBUK E D, ZOPH B, SHLENS J, et al. Randaugment: Practical automated data augmentation with a reduced search space[C]. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2020: 3008-3017.

# 常见问题

1. Semi-sklearn的接口和sklearn半监督学习模块的接口有什么不同？

sklearn的接口的fit()方法一般有X和y两项，无标注的X对应的标注y用-1表示。但是在很多二分类任务中，-1表示负类，容易冲突，因此Semi-sklearn的fit()方法有X,y和unlabeled_X三项输入。

2. DeepModelMixin模块如何理解？

这一模块主要是使深度学习与经典机器学习拥有相同的接口，并且为了便于用户更换深度学习种对应的组件，DeepModelMixin对pytorch进行了解耦。










