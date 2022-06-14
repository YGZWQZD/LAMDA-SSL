#  Introduction

Semi-sklearn is an useful and efficient toolbox for semi-supervised learning. At present, Semi-sklearn contains 30 semi-supervised learning algorithms, including 13 algorithms based on classical machine learning models and 17 algorithms based on deep neural network models. Semi-sklearn can be used to process four types of data: structured data, image data, text data and graph data, and can be used for three types of task: classification, regression and clustering. Semi-sklearn includes multiple modules such as data management, data transformation, algorithm application, model evaluation and so on, which facilitates the completion of the end-to-end semi-supervised learning process, and is compatible with the current popular machine learning toolkit scikit-learn and deep learning toolkit pytorch.

## Design Idea


The overall design idea of Semi-sklearn is shown in the figure. Semi-sklearn refers to the underlying implementation of sklearn. All Algorithms in Semi-sklearn have interfaces similar to that Algorithms in sklearn.The learners in sklearn all inherit the parent class Estimator. Estimator uses known data to build a model to make predictions on unknown data. There are usually two usual methods in Estimator: fit() and transform(). fit() method is an adaptation process using existing data to build a model, which corresponds to the training process in machine learning. transform() method is a transformation process using the fitted model to predict results of new instances, which corresponds to the predicting process in machine learning.

<div align=center>
<img width="500px"  src="./Imgs/Base.png" >
</div>



Learners in Semi-sklearn indirectly inherits Estimator in sklearn by inheriting the semi-supervised predictor class SemiEstimator. Data used for fit() method in sklearn usually includes samples and labels.However, in semi-supervised learning, labeled samples, labels and unlabeled samples are used in the training process of the model, so Estimator's fit() method is inconvenient directly used in semi-supervised learning algorithms. Although sklearn also implements two types of semi-supervised learning algorithms, self-training methods and graph-based methods, which also inherit the Estimator class, in order to use the interface of the fit() method, sklearn combines labeled samples and unlabeled data samples as samples input of fit() method, mark the labels corresponding to the unlabeled samples as -1 in labels input of fit() method. Although this processing method can be adapted to the Estimator interface, it also has limitations, especially in some binary classification scenario using -1 to indicate negative labels of labeled samples, which will conflict with unlabeled samples. Therefore, it is necessary to re-establish a new class SemiEstimator on the basis of Estimator for semi-supervised learning. There are three parts in the input of SemiEstimator's fit() method: labeled samples, labels and unlabeled samples, which better adapts the application scenario of semi-supervised learning. It doesn't require users combining data by themselves, and avoids the conflict between marks of unlabeled samples and labels of negative samples in binary classification. Compared with Estimator it's more convenient.


Semi-supervised learning is generally divided into inductive learning and transductive learning. The difference is whether the samples to be predicted is directly used as the unlabeled samples in the training process. Semi-sklearn uses two classes InductiveEstimator and TransductiveEstimator, which correspond to two types of semi-supervised learning methods: inductive learning and transductive learning respectively. InductiveEstimator and TransductiveEstimator both inherit SemiEstimator.

<div align=center>
<img width="500px"  src="./Imgs/LearningPattern.png" > 
</div>




In order to enable estimators to have corresponding functions for different tasks, sklearn has developed components (Mixin) corresponding to the scene for different usage scenarios of estimators. Estimators in sklearn often inherit both Estimator and corresponding components, so that they have the most basic fitting and prediction capabilities and also have the function of processing tasks corresponding to specific components. The key components include ClassifierMixin for classification tasks, RegressorMixin for regression tasks, ClusterMixin for clustering tasks, and TransformerMixin for data transformation, which are also used in Semi-sklearn.



In addition, different from sklearn framework commonly used in classical machine learning, deep learning often uses pytorch framework. There are lots of dependencies between the components of pytorch such as Dataset coupling with Dataloader, Optimizer coupling with Scheduler, Sampler coupling with BatchSampler, etc. In pytorch, there is no simple logic and interfaces like sklearn which causes great difficulty in integrating both classical machine learning methods and deep learning methods into a same toolkit. In order to solve this problem Semi-sklearn uses DeepModelMixin component to enable deep semi-supervised models developed based on pytorch to have the same interface and usage as classical machine learning methods. Deep semi-supervised learning algorithms in Semi-sklearn all inherit this component. DeepModelMixin decouples each module of pytorch, which is convenient for users to independently replace data loader, network structure, optimizer and other modules in deep learning without considering the impact of replacement on other modules. Deep semi-supervised learning algorithms can be called as easily as classical semi-supervised learning algorithms in Semi-sklearn.

<div align=center>
<img width="600px"  src="./Imgs/PytorchCoupling.png" > 
</div>


## Data Management

Semi-sklearn has powerful data management and data processing functions. In Semi-sklearn, a semi-supervised dataset can be managed by SemiDataset class. The SemiDataset class can manage three sub-datasets: TrainDataset, ValidDataset, and TestDataset corresponding to training dataset, validation dataset and test dataset in machine learning tasks respectively. The most basic classes for data management are LabeledDataset and UnlabeledDataset, which correspond to labeled data and unlabeled data in semi-supervised learning respectively. Training datasets often contains both labeled data and unlabeled data. Therefore, TrainDataset simultaneously manage a LabeledDataset and an UnlabeledDataset. 

Semi-sklearn designs two most basic data loaders: LabeledDataloader and UnlabeledDataloader for LabeledDataset and UnlabeledDataset respectively.Semi-sklarn uses TrainDataloader class to manage the two loaders in the training process of semi-supervised learning and adjust the relationship between the two loaders, such as adjusting the ratio of labeled data and unlabeled data in each batch. 

Semi-sklearn can process structured data, image data, text data, and graph data, which are four common data types in practical applications. Semi-sklearn uses four components corresponding to data types: StructuredDataMixin, VisionMixin, TextMixin, and GraphMixin. Every Dataset can inherit the component corresponding to its data type to obtain the data processing function in the component.

<div align=center>
<img width="600px"  src="./Imgs/Dataset.png" > 
</div>


## Data Transformation

Before using data to learn models and using models to predict labels of new data, it is usually necessary to preprocess or augment data, especially in the field of semi-supervised learning. To meet the needs of adding noise, the data transformation module of Semi-sklearn provides various data preprocessing and data augmentation methods for different types of data, such as normalization, standardization, MinMaxScale for structured data, Rotation, cropping, flipping for visual data, word segmentation, word embedding, length adjustment for text data, node feature standardization, k-nearest neighbor graph construction, graph diffusion for graph data, etc. All data transformation methods inherit TransformerMixin class from sklearn. Transformation method can be called using the interface of either sklearn or pytorch. For multi transformations in turn, both Pipeline mechanism in sklearn and Compose   mechanism in pytorch can be used. 

## Algorithm Usage

At present, Semi-sklearn contains 30 semi-supervised learning algorithms. There are 13 algorithms based on classical machine learning models, including generative method: SSGMM; semi-supervised support vector machine methods: TSVM, LapSVM; graph-based methods: Label Propagation, Label Spreading;  wrappers methods: Self-Training, Co-Training, Tri-Training; ensemble methods: Assemble, SemiBoost; semi-supervised regression method: CoReg; semi-supervised clustering method: Constrained K Means, Constrained Seed K Means. There are 17 algorithms based on deep neural network models, including Consistency regularization methods: Ladder Network, Pi Model, Temporal Ensembling, Mean Teacher, VAT, UDA; pseudo-label-based methods: Pseudo Label, S4L; hybird methods: ICT , MixMatch, ReMixMatch, FixMatch, FlexMatch; deep generative methods: ImprovedGAN, SSVAE; deep graph based methods: SDNE, GCN.

<div align=center>
<img width="1000px"  src="./Imgs/ClassicalSSL.png" >
</div>


<div align=center> 
<img width="1000px"  src="./Imgs/DeepSSL.png" > 
</div>


## Model Evaluation

Semi-sklearn provides different evaluation indicators for different tasks, such as accuracy, precision, recall for classification tasks, mean squared error, mean squared logarithmic error, mean absolute error for regression tasks and Davies Bouldin Index, Fowlkes and Mallows Index, Rand Index for clustering tasks etc. In Semi-sklearn, the evaluation method can be called after getting the prediction results directly passed to the model in the form of a python dictionary as a parameter.

# Quick Start

## Load Data

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

## Transform Data

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

## Use Pipeline Mechanism

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

## Train a Classical SSL Model

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

## Train a Deep SSL Model

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

## Search Params

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

## Train Distributedly

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

## Save and Load Model

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

# User Guide

## Algorithm

### Classical Semi-supervised Learning

#### Generative Model

Generative semi-supervised learning methods are based on generative models, which assume that data is generated from a potential distribution. Generative methods establish the relationship between samples and generative model parameters. In semi-supervised generative methods, labels are regarded as the latent variables of the model, The expectation-maximization (EM) algorithm can be used for maximum likelihood estimation.

##### SSGMM

SSGMM model was proposed by Shahshahani et al. SSGMM is a semi-supervised Gaussian mixture model. It is assumed that data is generated by a Gaussian mixture model, that is, the marginal distribution of samples can be expressed as the result of mixing several Gaussian distributions together, and each Gaussian distribution is given a weight. SSGMM maps the feature distribution of each class to a Gaussian mixture component. For each labeled instance, the Gaussian mixture component corresponding to its class is known. For each unlabeled instance, the Gaussian mixture component is represented by a probability distribution, and it can be classified into the class corresponding to the Gaussian mixture component with the highest probability. SSGMM assumes that the samples obey the assumption of independent and identical distribution, and its likelihood function is the product of joint probabilities of all labeled examples and marginal probabilities of all unlabeled data samples. The maximum likelihood estimation is used to maximize the likelihood function to get the parameters of the generative model including the variance, mean, and weight of each part of the Gaussian mixture model with the highest probability of co-occurrence of labeled data and unlabeled data. Since this method has unobservable hidden variables corresponding labels of unlabeled samples, it cannot directly get maximum likelihood parameters, so SSGMM adopts the EM algorithm to solve this problem. The EM algorithm is divided into two steps. In E step, conditional distribution or expectation of the unobserved data are obtained according to the current parameters and the observable data. In the SSGMM model, this step uses the Bayesian formula to get the conditional distribution of the labels of the unlabeled samples according to the observed samples and the parameters of the current model. The M-step makes a maximum likelihood estimation of the model parameters according to the value of the currently observed variable and the expectation or probability distribution of the latent variable, that is, the original The hidden variables are unknown and the maximum likelihood estimation cannot be performed. After the E step, the expectation or conditional probability distribution of the hidden variables is obtained, and the maximum likelihood estimation becomes feasible. In the SSGMM model, this step uses the observed labeled samples and labels, unlabeled samples and class conditional distributions obtained in step E update the parameters of the Gaussian mixture model. Step E and step M are carried out alternately in an iterative form until convergence, which can realize the simultaneous use of labeled data and unlabeled data to train a Gaussian mixture model and the classifier based on this Gaussian mixture model can be obtained through the Bayesian formula.
<!-- 
#### <font color=purple size=32>Semi-supervised Support Vactor Machine</font> -->
#### Semi-supervised Support Vactor Machine

Support vector machine is one of the most representative algorithms in the field of machine learning. This class of algorithms treats the binary classification problem as finding a suitable partitioning hyperplane in the feature space. In the case of linear separability, among all the hyperplanes that can complete the correct classification, the optimal dividing hyperplane should be located in the middle of the samples from different classes, which can improve the robustness of the model for predicting unknown samples. In each class, the sample closest to the hyperplane is called the support vector. Support vectors of different classes are equidistant from the hyperplane. The purpose of the support vector machine algorithm is to find the hyperplane closest to its corresponding support vectors. However, in real tasks, there are often no hyperplanes that can correctly divide all training samples. Even if there are, it is most likely due to overfitting. Therefore, a class of support vector machine methods introduces Soft Margin mechanism , which allows the hyperplane to not necessarily classify all samples correctly, but adds a penalty for misclassifying samples in the optimization objective.

Semi-supervised support vector machine is a generalization of the support vector machine algorithm in the field of semi-supervised learning. The semi-supervised support vector machine introduces a low-density assumption, that is, the learned hyperplane not only needs to separate the classifications as much as possible based on the labeled samples, but also needs to pass through the low-density regions of the distribution of all samples as much as possible. It make reasonable use of the unlabeled samples. Semi-sklearn includes two semi-supervised SVM methods: TSVM and LapSVM.

<!-- ##### <font color=blue size=16>TSVM</font> -->
#### TSVM

TSVM  model was proposed by Joachims et al. TSVM is the most basic transductive semi-supervised support vector machine method. TSVM needs to infer labels of unlabeled samples and find a dividing hyperplane that maximizes the distance from support vectors. Since the labels assigned to the unlabeled samples are not necessarily real labels, the unlabeled samples and the labeled samples cannot be treated the same in the initial stage of training. $C_l$ and $C_u$ are magnitudes of the penalty reflecting the importance attached to the labeled samples and the unlabeled samples. Since the number of possible situations where labels of all unlabeled samples may appear increases exponentially with the increase of the number of unlabeled samples, it is impossible to determine the labels of unlabeled samples in an exhaustive way to find the global optimal solution. The optimization goal of TSVM is to find an approximate solution using an iterative search method. Firstly an SVM is trained based on the labeled samples, and the SVM is used to predict the labels of unlabeled samples. Then $C_l\ll C_u $ is initialized, and the iteration is started. In the iterative process, all the samples are used to solve the new hyperplane. TSVM algorithm continuously finds pairs of unlabeled heterogeneous samples that may be mispredicted and exchange labels and retrain the SVM model until no more qualified pairs can be found. The importance of unlabeled samples is increased by doubling the value of $C_u$ in each iteration. Iteration continues until $C_u$ is equal to $ C_l$. Finally, the prediction results of the unlabeled samples obtained by the final SVM model and the transductive process is completed.

<!-- ##### <font color=blue size=56>LapSVM</font> -->
##### LapSVM
 
LapSVM was proposed by Belkin et al. LapSVM is based on the manifold assumption. The classical SVM algorithm seeks to maximize the margin between the dividing hyperplane and support vectors, which only considers the distribution of samples in the feature space. However, in practical applications, samples in high-dimensional space are often distributed on low-dimensional Riemannian manifolds. Classical SVM based on original feature space tends to ignore the essential characteristics of samples. LapSVM adds a manifold regularization term to the optimization objective of SVM for learning the essential distribution of samples. LapSVM builds a graph model using all samples, obtains the weight matrix of the graph model through the similarity between the features of the samples and calculates its Laplace matrix. The Laplace regularization term guides the predicted results of the adjacent samples in the graph to be as consistent as possible. Unlike TSVM, LapSVM only penalizes the misclassification of labeled samples, but uses all samples when building a graph model, so that unlabeled samples participate in the learning process by the distribution of samples on the manifold.

#### Graph Based Method

The graph-based semi-supervised learning method represents the data set as a graph model with samples as nodes and relationships among samples as edges. In semi-supervised learning, there are labeled data and unlabeled data, so some points have labels, while others haven't. Graph-based transductive semi-supervised learning can be regarded as the process of label propagation or spreading in the graph.


##### Label Propagation

Label Propagation was proposed by Zhu et al. Label Propagation uses samples as nodes, and the relationship between the samples as edges. The graph can be constructed fully connected or based on k-nearest neighbors. The purpose of Label Propagation algorithm is to propagate the labels from labeled data to unlabeled data through the graph. The optimization goal of Label Propagation is the Laplacian consistency, that is the weighted mean square error of the difference of labels between pairs of adjacent nodes. Since the labels of labeled samples are fixed, Label Propogation only needs to solve the labels of unlabeled data to minimize the optimization goal. So the model's predictions for adjacent points on the graph should be as consistent as possible. Label Propagation makes the partial derivative of the optimization objective to the labels of unlabeled samples to be 0 and the optimal solution can be obtained. It has been proved that this closed-form optimal solution obtained by direct calculation is consistent with the final convergence result of infinite and continuous iterations. An accurate solution can be obtained through direct derivation, without simulating the process of label propagation and performing multiple iterations for convergence, which is the advantage of Label Propagation over other graph based semi-supervised learning methods.

##### Label Spreading

 Label Spreading was proposed by Zhou et al. Different from Label Propagation algorithm in which the labels of the labeled samples are fixed during the spreading process to protect the influence of the real labels on the model, Label Spreading penalizes misclassified labeled samples rather than banning it completely. For the existence of data noise, Label Prapogation has certain limitations and does not performs well. An labels in Label Propagation algorithm can only flow to unlabeled nodes, which may block some paths that need to be propagated through labeled nodes, which limits the propagation of information in the graph. Label Spreading algorithm enables labels to be broadcast to all adjacent nodes to improve this problem.  There are two optimization goals for Label Spreading. The first is the same as that of Label Propagation, but there is no restriction that the model's prediction results for labeled samples must be equal to its true label. The second is the prediction loss for labeled data with a penalty parameter as its weight. Due to different optimization goals, Label Propagation has a closed-form solution, while Label Spreading needs to be solved iteratively. In each iteration, a trade-off parameter is required to weight spreading results and initial labels of samples as the current prediction results.


#### Wrapper Method

Unlike other semi-supervised learning methods that use unlabeled data and labeled data to train a learner together, Wrapper methods is based on one or more wrapper supervised learners. These supervised learners can often only process labeled data, so they are always closely related to pseudo-labels of unlabeled data. In addition, unlike other methods that require a fixed learner, wrapper methods can arbitrarily choose its supervised learner, which is very flexible and facilitates the extension of existing supervised learning methods to semi-supervised learning tasks. It has strong practical value and Low application threshold.

##### Self-Training

Self-Training was proposed by Yarowsky et al. The Self-Training is the most classical wrapper method. It is an iterative method. Firstly, a supervised classifier is trained with labeled data. Then in each round of iteration, the current learner is used to make prediction on unlabeled samples to obtain its pseudo labels. Unlabeled samples with their pseudo labels whose confidence higher than a certain threshold are combined  with labeled dataset to form a new mixed dataset. Lately a new classifier trained on the mixed data set is used in the next iteration process. The training pocess of Self-Traing is so convenient that any supervised learner which provides soft labels can be used. Self-Training provides a basis for subsequent research on other wrapper methods.


##### Co-Training

Co-Training was proposed by Blum et al. In Co-Training, two basic learners are used to cooperate with each other to assist each other's training. For unlabeled samples with high confidence on one learner, Co-Training will pass them and their pseudo-labels to another learner, through this form of interaction, the knowledge learned on one learner is passed on to another learner. Due to the difference between the two base learners, their learning difficulty for same samples is different. Co-Training effectively takes advantage of this difference, so that a learner can not only use its own confident pseudo-labels, but also use another learner's confident pseudo-labels to increase the utilization of unlabeled data. Finally the two learners are ensembled to be used for prediction. In order to make the two base learners have a certain difference, Co-Training adopts the multi-view assumption, that is, models trained based on different feature sets should have the same prediction results for the same samples. Co-Training divides the features of the samples into two sets, which are used as observations of the samples from two different views. The training data of the two learners in the initial state are only labeled data in different views. In the initial state, the training datasets of the two learners are only labeled dataset in different views. In the iterative process, Unlabeled samples with high pseudo-label confidence on one learner are added to the training datasets of two learners at the same time, which will be used for the next round of training. The iteration continues until the predictions of both learners no longer change.

##### Tri-Training

Tri-Training is proposed by Zhou et al. Because multi-learner training methods such as Co-Training must require differences between basic learners, such as different data views or different models. However, in practical applications, there may only be a single view of data, and artificially cutting the original data will lose the information of the relationship between the features in different views. The division method for views requires expert knowledge because wrong division may lead to serious performance degradation. The use of different models for co-training requires setting up multiple supervised learners. Considering that the advantage of wrapper methods over other semi-supervised learning methods is that the supervised learning algorithm can be directly extended to semi-supervised learning tasks, so there is often only one supervised learning algorithm in the scenarios where the wrapper methods are used. Designing additional supervised learning algorithms loses the convenience of wrapper methods. Tri-Training solves this problem from the perspective of data sampling. Only one supervised learning algorithm is used, and the dataset is randomly sampled multiple times to generate multiple different training datasets to achieve the purpose of learning to get multiple different models. In other wrapper methods, the model's confidence in unlabeled samples is used to decide whether to incorporate unlabeled data into the training data set. However, in some cases, the model may be overconfident in the misclassification, resulting in a large deviation. Tri-Training uses three base learners for training. During the training process of one base learner, for unlabeled samples, the other two base learners can be used to judge whether they should be added in the training dataset. If in one iteration, the common prediction error rate of the two learners is low and they have the same prediction result for the unlabeled sample, the unlabeled sample and its pseudo-label are more likely to produce positive effect to the current training base learner. Using the prediction consistency of the two models to determine whether to use unlabeled samples is more robust than using the confidence of only one model. In addition, in other wrapper methods, the selected unlabeled samples will always exist in the training dataset, which may cause the unlabeled samples that are mispredicted to have a lasting impact on the learner and can never be corrected. In Tri-Training, the unlabeled samples and their pseudo-labels used in each iteration of the algorithm are reselected. Tri-Training has a solid theoretical foundation and adds restrictions on the use of unlabeled data in each iteration based on the theoretical foundation. Strict constraints on the use of unlabeled data greatly increase the security of the semi-supervised model, which can effectively alleviate the problem of model performance degradation caused by erroneously using unlabeled data.

#### Ensemble Method

In the field of machine learning, the use of a single learner is likely to cause high deviation or variance due to underfitting or overfitting, resulting in insufficient model generalization ability. Ensemble learning combines multiple weak learners, which not only improves the model's ability to represent the hypothesis space, but also reduces the impact of errors caused by a single learner and improves the reliability of the model. In the field of semi-supervised learning, due to the addition of unlabeled data, using a single learner to set pseudo-labels for unlabeled samples further exacerbates the instability of a single learner and has a stronger reliance on effective ensemble learning methods.

##### Assemble

Assemble is proposed by Bennett et al. Assemble is an extension of AdaBoost method in the field of semi-supervised learning. The Boosting method is an important method in ensemble learning. This method samples the data set through the prediction effect of the current ensemble learner. The sampling process will pay more attention to the samples whose results of the current ensemble learner is not good. This strategy enables the model to pay more attention to the samples with poor learning effect of the current ensmeble learner in each round of new weak learner learning process, and continuously improve the generalization ability and robustness of the model. AdaBoost is the most representative method in the Boosting methods. This method adaptively adjusts the sample weight according to the difference between the prediction results and the samples' real labels. The weak learners with higher accuracy have higher ensemble weights. ASSEMBLE promotes the AdaBoost method in the field of semi-supervised learning by adding pseudo-labels to unlabeled data. In the initial stage, the pseudo-labels of unlabeled samples are the labels of the closest labeled samples and the unlabeled data and labeled data have different weights. In the iterative process, the pseudo-labels of unlabeled data are updated to the prediction results of the ensemble learner in each round. As the iteration progresses, the effect of the ensemble learner is getting better and better and the pseudo-labels are more and more accurate, which further have a more beneficial impact on the ensemble learner.

##### SemiBoost

SemiBoost are proposed by Mallapragada et al. Unlike Assemble, which only uses the difference between the prediction results of the model and the real labels or pseudo-labels to weight samples and does not consider the relationship between samples, SemiBoost is based on graph semi-supervised learning method, which points out that the similarity between samples also should be taken into consideration and a larger weight should be set for the samples with high similarity in feature space and high inconsistency in prediction results to other samples. The generalization ability and robustness of model are improved. Unlabeled samples play a greater role in this process. SemiBooost learns a new weak learner in each round of iteration. Its optimization objective consists of two items. The first item punishes the discrepancy between pseudo-labels of unlabeled samples and real labels of labeled samples which uses the similarity in feature space as weights. It is close to the effect of Label Propagation so that the model can obtain pseudo-labels of unlabeled samples according to the graph structure. The second term penalizes the prediction between unlabeled samples which uses the similarity within the unlabeled samples as weights. Te second item alleviates the impact of noise to the model.

#### Semi-supervised Regression

Most of the current semi-supervised learning algorithms are designed for classification tasks and cannot be naturally extended to regression tasks. Only a few works are aimed at semi-supervised learning regression tasks. It is because in regression tasks, reasonable assumption are more difficult to be made compared to classification tasks. At present, there are few research results in this field and the semi-supervised regression task still has great demand and application value in real scenarios.

##### CoReg

CoReg is proposed by Zhou et al. CoReg introduces the Co-Training algorithm into regression tasks. In wrapper methods originally used for classification, it is often assumed that samples with higher confidence will have a more positive impact on subsequent training. Some unlabeled samples and their pseudo-labels are selected as training data according to the confidence. However it is difficult to get the confidence in the regression task. CoReg solves this problem, thereby applying the Co-Training algorithm to regression tasks. CoReg uses the kNN model as the type of base learner. For two base learners, in order to maintain the difference between them, different orders are used to calculate the Minkowsky distance between samples in the k-nearest neighbor model. In order to measure the confidence of the model on the samples, for each unlabeled sample, the model first predicts the real-valued pseudo-label. CoReg then combines its pseudo-label with all the samples participating in the training process to retrain a learner and uses the mean squared error loss to evaluate to evaluate the impact of the sample on its k nearest neighbors. If the mean square error of these k-nearest neighbor nodes decreases, it means that adding the unlabeled sample is more likely to have a positive impact on subsequent training. Therefore, CoReg uses the difference between the mean square error before and after adding the unlabeled sample as the evaluation standard of confidence. The unlabeled sample with the highest confidence and its real value pseudo-label are added to the training dataset of another learner, thus completing the training process of Co-Training.

#### Semi-supervised Cluster

Unlike semi-supervised classification and semi-supervised regression tasks, which use unlabeled data to assist the process of supervised learning, semi-supervised clustering tasks introduce supervision information to assist the process of unsupervised learning. Supervision information is not necessarily labeled data, but may also be other knowledge related to real labels. Due to the difference of supervised information, various semi-supervised clustering methods have also been proposed.

##### Constrained k-means

Constrained k-means was proposed by Wagstaff et al. Based on k-means clustering algorithm, the algorithm introduces constraints called Must Link and Connot Link as supervision information. The Must Link constraint restricts that some samples must belong to the same cluster. The Connot Link constraint restricts that some samples must belong to different clusters. There is transfer mechanisms in constraints. For example, if A and B must be linked and B and C must be linked, then A and C must be linked and if A and B must be linked and B and C can not be linked, then A and C can not linked. K-means algorithm assigns a sample to the cluster whose cluster center is closest to the sample. Similarly, Constrained k-means also give priority to the cluster whose center is closest to the sample, but the difference is that Constrained k-means algorithm should judge whether the Must Link and Cannot Link constraints are violated between the sample and the samples which are already in the cluster. If violated, Constrained k-means will reconsider the next eligible cluster. If all clusters fail to satisfy the constraints, a warning of clustering failure is issued and different cluster centers are needed to be selected randomly to reinitialize the process.

##### Constrained Seed k-means

Constrained Seed k-means was proposed by Basu et al. This algorithm is different from Constrained k-means, which uses Must Link and Connect Link constraints as supervision information, but directly uses labeled data as supervision information. Since there are some labeled data, the cluster center can be calculated directly on the labeled dataset, which effectively alleviates the cluster instability caused by the randomness of the initial cluster centers selection. The number of classes of the labeled dataset can be used as the number of clusters in the clustering algorithm, which avoids the bad clustering results caused by unreasonable k value selection. Unlike k-means algorithm, in which all samples are judged to which cluster they should belong in the iterative process, Constrained Seed k-means algorithm only updates the cluster labels of unlabeled data. For labeled samples, their cluster labels are fixed with their real labels and not change as the change of cluster centers. The clusterer is more reliable when using labeled data to participate in the clustering process, which alleviates the blindness of unsupervised clustering and effectively reduces the large gap between the clustering results and the real labels of the samples. It also alleviates the instability caused by randomness.

### Deep Semi-supervised Learning

#### Consistency Regularization

Deep learning methods guide the direction of model training by setting the loss function with gradient descent. Consistency regularization methods are based on the assumption of consistency, which assumes if a certain degree of disturbance is added to samples, the prediction results should be consistent with the previous. These methods often introduces a consistency regularization term into the loss function which enables unlabeled samples to participate in the model training process to improve the robustness of the model to noise

##### Ladder Network

LadderNetwork was proposed by Rasmus et al. This method adopts an autoencoder structure, in which the outputs of the last layer of the encoder are soft labels. The LadderNetwork adopts two encoding methods, the first is the classical encoder without noise, that is, and the second is the encoder with noise, which add noise to inputs of each layer of the classical encoder.   LadderNetwork firstly performs noise encoding and non-noise encoding on the samples respectively and obtains the noisy representation and the non-noisy representation of each layer. Then the decoder is used to decode the noisy encoding result, and the noisy decoding representations of each layer are obtained. Finally, mean square error(MSE) loss is used to calculate the inconsistency between the non-noisy encoded representation and the noisy decoded representation at each layer, including the original input data as the zeroth layer. The previously determined weights are used to determine the weights of inconsistency of each layer. Hierarchical inconsistencies are weighted as an unsupervised loss function, thereby improving the robustness of model. The consistency regularization of LadderNetwork uses the noisy encoded representation as a bridge to penalize the inconsistency between the non-noisy encoded representation and the noisy decoded representation. On the one hand, an auto-encoder can be obtained to make the representations of the encoder and the decoder consistent at all levels. On the other hand, the hidden layer representations keep consistent regardless weather noise is added, which makes the model can against disturbances.

##### UDA

UDA was proposed by Xie et al. Unlike LadderNetwork, UDA only perturbs the input samples instead of all inputs of hidden layers. And UDA does not necessarily use Gaussian noise for perturbation, but may use various data augmentation methods. Compared with Gaussian noise, data augmentation used by UDA, such as image rotation or text replacement have a greater impact on the data, which can further improve the robustness of the model. UDA performs data augmentation on the unlabeled samples and then compares the prediction results before and after the augmentation. The mean square error loss is used to calculate the consistency regularization term as the unsupervised loss.

##### Pi Model

Pi Model was proposed by Laine et al. Unlike UDA, which augments the unlabeled data once and compares the prediction results before and after the augmentation and calculates the consistency regular term. Pi Model augments the data twice randomly and respectively uses the results of the two augmentations as inputs of the neural network model to get prediction results. The inconsistency of the prediction results are used as the unsupervised loss. Due to the randomness of the augmentation process, the two augmentations of this method will obtain two pieces of samples that are semantically similar but may have slightly difference in features. Through the consistency regularization, the model can produce similar prediction results for different augmentations with a certain range. 

##### Temporal Ensembling

Temporal Ensembling are proposed by Laine et al. This method makes some improvements to Pi Model. In Pi Model, for each unlabeled sample, Pi Model needs to perform two augmentations and calculate the inconsistency of their prediction results, which brings a large consumption of computing power. Temporal Ensembling method changes one of the pseudo-label predictions to exponential moving average(EMA) of historical pseudo-labels, which is a weighted average of historical results. The weights of pseudo-labels decay exponentially in each round. This ensemble method effectively preserves the historical pseudo-labels information and get unsupervised loss by calculating the consistency between the current pseudo-label and the ensemble of historical pseudo-labels. Tthe historical ensemble is updated at the end of each epoch. EMA guarantees the robustness of the model. It avoids the model being overly affected by a single round of prediction and slows down the model’s forgetting speed of historical information. Temporal Ensembling only needs to augment and predict once for each sample in each round. Historical information can be maintained with only one weighted average calculation, which greatly reduces computing power consumption compared to Pi Model.

##### Mean Teacher

Mean Teacher was proposed by Tarvainen et al. This method relies on the idea of ​​knowledge distillation, where the prediction results of the teacher model are used as pseudo-labels to train the student model to ensure the consistency of the prediction results of the teacher model and the student model, thereby distilling knowledge from a more complex model to a simpler one. The purpose of the classical knowledge distillation method is to simplify the model, but the purpose of Mean Teacher is to make unlabeled data participate in the learning process and improve the robustness of the model. Therefore, the teacher model is not a complex model, but performs exponential moving average on the parameters based on the student model, which reduces the computational cost compared to the classical knowledge distillation method. Temporal Ensembling method performs EMA on the prediction results of each round, but the overall historical information only can be updated at the end of each round. Especially for large data sets, the historical information cannot be updated in time. Different from Temporal Ensembling, Mean Teacher uses EMA for model parameters and updates the historical information of model parameters in time after each batch of training. Mean Teacher is more flexible and general because it effectively solves the problem of untimely update and utilization of historical information. 

##### VAT

VAT was proposed by Miyato et al. Different from the methods of adding random noise to the data, the idea of ​​VAT is to add adversarial noise to the data, so that the worst performance of the model can be better when the data is affected by noise within a certain range, which corresponds to the zero-sum game in game theory and Min-Max problem in optimization. For classical supervised adversarial algorithms, the cross-entropy loss between the real labels and prediction results is usually used as the goal of adversarial optimization. The noise that maximizes the loss for the current model and data is obtained through the inner layer optimization. The outer layer optimization is used to obtain the model parameters which minimizes the loss. Inner and outer optimization are alternately performed, so that the model can not perform too bad in the worst case when dealing with data noise. The outer optimization is to optimize the model parameters, which is often carried out by gradient descent, while the inner optimization is optimized for data noise, in which there is no closed-form solution. The inner optimization is not suitable to use gradient optimization and it is necessary to approximate the optimal noise. Linear approximation is often used in classical supervised adversarial algorithms. It firstly predicts on the clear data and calculate the value of the loss function. Then it carries out the gradient backward to obtain the gradient. Finally it takes the product of the normalized gradient and the noise upper bound as the adversarial noise. Different from classical supervised adversarial algorithms, VAT needs to solve the problem in semi-supervised scenarios where loss of unlabeled data cannot be calculated supervisely and then adversarial noise cannot be obtained by gradient. In order to solve this problem, VAT adopts the consistency strategy. It changes the supervised loss to the consistency loss which uses the model to predict on the clear data and the noisy data respectively to obtain the clear pseudo-labels and the noisy pseudo-labels. Then it calculates the consistency between them. In VAT, linear approximation cannot be used for the inner optimization on unlabeled data because it is necessary to calculate the classification loss with real labels and VAT replaces real labels with pseudo-labels resulting in the gradient returned is always 0. So VAT uses second-order Taylor approximation instead of linear approximation, so the problem of computing against noise is transformed into the problem of computing the principal eigenvectors of the Hessian matrix of the loss function for noise. When the noise of data is d-dimensional, the time complexity of calculating the eigenvector of Hessian matrix is O\left(d^3\right). In order to solve the problem of excessive computational complexity, VAT adopts power iteration method to solve the approximate matrix eigenvectors, which randomly samples the approximated eigenvectors and continuously multiply the matrix and the current approximated eigenvectors to obtain new ones. Continuously performing this process can consume less time. In order to further avoid the direct calculation of the Hessian matrix, VAT adopts the Finite Difference method to approximate the product of the matrix and the approximate eigenvector. Compared with other methods based on consistency regularity, the use of anti-noise in the VAT method can further improve the robustness of the model than random noise and avoid excessive interference of randomness on the experimental results because the performance in the worst case has a better theoretical basis. VAT avoids excessive additional computational overhead through approximation methods when calculating adversarial noise and solves the dilemma that supervised adversarial algorithms cannot be directly applied to unlabeled data.

#### Pseudo Labeling

Methods based on pseudo-labeling make unlabeled data affect the learning process by assigning pseudo-labels to unlabeled data. Since the confidence levels of the model for different samples are different, the method based on pseudo-labels usually takes samples with higher confidence and their pseudo-labels to participate in the training process.

##### Pseudo Label

Pseudo Label was proposed by Lee et al. This method is the most basic pseudo-labeling method. Its loss function includes two items supervised loss and unsupervised loss, both of which are cross-entropy loss functions. For unlabeled data, the Pseudo Label performs softmax operation on the output of the neural network to obtain the confidence of classification. Pseudo Label takes the category with the highest confidence as the pseudo-label of the sample and uses the pseudo-label to calculate the cross-entropy loss. In addition, in each round, not all unlabeled samples will participate in the training process. They participate in the training process only when the confidence of them in this round are greater than the set threshold. Pseudo Label also sets hyperparameters to control the proportion of unsupervised loss and supervised loss and adopts a warmup mechanism. At the beginning of training, the proportion of unsupervised loss is low and as the training goes on, the proportion is increasing.

##### S4L

S4L was proposed by Beyer et al. This method uses self-supervised technology. The basic idea is that unlabeled data cannot directly participate in the training of classifiers, but self-supervision can be used to affect the representation layer, so that the model can learn better hidden layer representations. This method is mainly used for image data. One of 0^\circle, 90^\circle, 180^\circle, and 270^\circle is randomly selected as the degree to rotate the image, and the angle is used as a pseudo-label. A neural network model can be trained to classify angles. Although the final classification layer of the neural network is different from the target task, the learned hidden layer representation is helpful for learning the real task. For labeled data, S4L uses two labels, a pseudo label representing the degree of rotation and a real label for the target task. S4L uses two classification layers for labeled data, one of which is the degree classifier shared with unlabeled data, and the other is the true classifier for the target task, and both classification layers share the same hidden layer. Through the above methods, S4L enables the model to learn better representations while training the self-supervised classifier, thereby improving the classification effect of the model on the target task. Different from the pre-training and fine-tuning paradigms, S4L does not need to train the model in advance, but can process labeled data and unlabeled data at the same time to promote each other. Labeled data also participates in the self-supervised learning process of S4L. S4L can also be generalized to other types of data, and corresponding self-supervised training methods need to be adopted.

#### Hybird Method

There are often no conflicts among different semi-supervised learning techniques. Many commonly used semi-supervised learning algorithms are not limited to using only one type of techniques, but combine techniques such as consistency and pseudo-annotation et al and use their own strengths to generate new hybrid methods. Hybrid methods can leverage the advantages of different techniques simultaneously to achieve better training results.

##### ICT

ICT was proposed by Verma et al[25]. The full name of  ICT is Interpolation Consistency Training. The data and prediction results are linearly interpolated by Mixup [34] which is a data augmentation method. ICT introduces unlabeled data into the training process by using the consistency between the model's predictions on the interpolated samples and the interpolation of the model's predictions on the original data. Mixup generates a parameter which means mixing ratio from the Beta distribution, and linearly interpolates two samples using the ratio parameter to obtain the mixed sample. The loss function of ICT is divided into two parts: supervised loss and unsupervised loss. The supervised loss is calculated by the cross entropy function and the unsupervised loss is calculated by the interpolation consistency. For each batch of data, ICT firstly samples a mixing parameter according to the Beta distribution. Then ICT randomly scrambles the batch of samples, and mixes the scrambled batch data with the unscrambled batch data in proportion to the mixing parameter. The model predicts on the unscrambled batch data and the mixed batch data to get the unscrambled prediction results and the mixed prediction results. Finally ICT linearly interpolates the unscrambled prediction results and the scrambled prediction results with the same mixing parameters as the samples and takes the inconsistency as the unsupervised loss. For mixed unlabeled data, ICT makes the soft labels output by the model close to the mix of pseudo-labels and combines consistency technology with pseudo-label technology to make the model more robust.

##### MixMatch

MixMatch was proposed by Berthelot et al. This method also uses Mixup method, but unlike ICT which only mixes unlabeled data samples, MixMatch mixes both labeled data and unlabeled data. MixMatch firstly augments the unlabeled data multiple times and makes multiple predictions. By averaging and sharpening the results of multiple predictions, the pseudo-labels of the unlabeled data are obtained. Multiple augmentations make the pseudo-labels more reliable. Sharpening the pseudo-labels reduces the entropy of the label distribution, allowing the classification boundaries to pass through the low-density regions of the samples as much as possible. Then MixMatch combines and shuffles the labeled data set and the unlabeled data set to form a new mixed data set. The same amount of samples as the original labeled samples are taken out from the mixed data set to form a new labeled data set by Mixup and the remaining samples in the mixed data set forms a new labeled data set by Mixup too. Finally, MixMatch predicts on the new labeled data set and the new unlabeled data set respectively. It uses the prediction results of the new labeled data set to calculate the cross entropy as the supervised loss and uses the new unlabeled data set to calculate the mean square error as the unsupervised loss. The two terms are combined by a weight parameter. Different from other methods which calculate the loss of labeled data and unlabeled data separately, MixMatch combines, shuffles, and re-partitions labeled data set and unlabeled data set, which reduces the risk of model performance degradation due to wrong pseudo-labels. MixMatch is helpful to use real labels to assist the training of unlabeled data and guide the correct training direction of unlabeled consistency which not only ensures the original robustness of the consistency regularization, but also prevents the model from excessive target deviation due to the inconsistency between pseudo-labels and real labels.

##### ReMixMatch

ReMixMatch was proposed by Berthelot et al. ReMixMatch is an improved version of MixMatch. It introduces two techniques: distribution alignment and augmented anchoring. The purpose of distribution alignment is to make the pseudo-labels predicted by the model for unlabeled data have the same marginal probability distribution as the real labels of labeled data. In deep learning, the label distribution of the labeled data and the pseudo-label distribution of the unlabeled data are different because the model's predictions are often biased towards the categories which have more samples and the use of a sharpening operation reduces the entropy of the label distribution to force the classification boundaries to pass through low-density regions as much as possible. There is an unfair phenomenon among categories in the pseudo-labels of data and the distribution alignment technology effectively alleviates this problem. The distribution alignment technology calculates the true label distribution of the labeled data. In each batch of training, the soft label distribution is calculated. For the soft label of a sample, ReMixMatch multiplys it by the ratio of the real label distribution and the current batch soft label distribution to obtain the aligned soft label and sharpens the aligned soft label to obtain the pseudo label of the sample. Augmented anchoring is to adapt the model to stronger data augmentation. For supervised learning methods, applying stronger data augmentation to the data can further improve the generalization ability of the model because no matter whether strong or weak augmentation is applied to the sample, its label will not change. In semi-supervised learning, pseudo-labels are often obtained from the prediction results on unlabeled data by the model. The pseudo-labels will change with the form of data augmentation. If a strong augmentation is applied to the samples, it is easy to make the pseudo-labels deviate too much from the real labels. It makes MixMatch incompatible with strong data augmentation methods. By introducing augmented anchoring technology, ReMixMatch performs weak data augmentation on unlabeled samples. The model predicts for weakly augmented unlabeled samples to get pseudo-labels and fixes them as "anchors", so that no matter what kind of data augmentation is performed on the unlabeled data in the future, the pseudo-labels will not change. ReMixMatch performs one weak data augmentation and multiple strong data augmentation on the unlabeled data, and uses the model's prediction results for the weakly augmented data as pseudo-labels after alignment and sharpening. The augmented dataset composes a larger unlabeled dataset. ReMixMatch uses the same strategy as MixMatch to combine, shuffle and re-partition the labeled and unlabeled datasets. In addition, the loss function of ReMixMatch is quite different from that of MixMatch. The supervised loss and unsupervised loss of ReMixMatch are both calculated by cross entropy and different from MixMatch's loss function which only includes supervised loss and unsupervised loss, ReMixMatch adds two items. Although Mixup makes the model have better generalization performance, only using the data after Mixup may ignore some information of the data set before Mixup, so ReMixMatch takes one out of multiple augmented data sets before Mixup and uses it to calculate the unsupervised loss of pre-Mixup dataset as the third term of the loss function. ReMixMatch also draws on the self-supervised strategy of S4L. Samples from the augmented dataset are randomly rotated and their rotation angles are predicted to promote the learning of the hidden layer of the model. The cross-entropy loss for classifying the rotation angle is used as the fourth term of the loss function. ReMixMatch integrates multiple techniques in a more complex framework that not only combines the strengths of each method, but is more general because of its comprehensiveness.

##### FixMatch

FixMatch was proposed by Sohn et al. FixMatch also uses strong data augmentation and weak data augmentation. Unlike ReMixMatch, which uses augmented anchoring to fix pseudo-labels of unlabeled data by weak data augmentation. FixMatch pays more attention to the consistency of prediction results between weakly augmented data and strong augmented data. Similar to ReMixMatch, FixMatch also obtains hard pseudo-labels according to the prediction results of the model for weakly augmented data. After that, FixMatch augments the unlabeled data strongly to obtain the prediction results. FixMatch only uses the unlabeled data with which the model is confident for training using a threshold parameter. Only when the confidence is greater than the threshold parameter, the data will participate in the training process. FixMatch calculates the cross-entropy using the pseudo-labels obtained by the model for weakly augmented samples and the prediction results obtained by the model for strong augmented samples as unsupervised loss. Fixmatch combines the unsupervised loss and the supervised loss by a weight parameter as the final loss.

##### FlexMatch

FlexMatch was proposed by Zhang et al[29]. FlexMatch is an improvement version of FixMatch and focuses on solving the unfair phenomenon between categories in semi-supervised learning. FixMatch selects unlabeled samples with high confidence and their pseudo-labels according to fixed threshold to participate in the training process. But sometimes although the original The dataset is class-balanced, due to the different learning difficulty of each class, using a fixed threshold for selecting will cause some classes which are difficult to learn to be less used in the training process than which are easy to learn. The model has lower confidence in the samples whose classes are more difficult to learn, which further exacerbates the class imbalance of the unlabeled data participating in the training. This unfairness forms a vicious circle and causes Matthew effect in Fixmatch. This unfairness forms a vicious circle, resulting in the Matthew effect, which causes the model to learn less and less well for the categories which are hard to learn. Therefore, different selecting criteria should be used for different categories to alleviate the class imbalance caused by different learning difficulties. FlexMatch uses dynamic threshold on the basis of FixMatch. It sets a lower confidence threshold for the classes that are more difficult to learn. One of the most basic methods is to set a validation dataset thresholds according to the accuracy rates on the validation dataset. However, since the labeled training data is relatively scarce, and the verification accuracy of the model is continuously updated during the training process, it will cause a large computational. Therefore, FlexMatch adopts the method of approximately evaluating the accuracy. Flexmatch firstly counts the number of times for each class that the class is consider as the pseudo-label and the confidence is greater than the threshold respectively for each batch of unlabeled data. After that, the statistics of different categories are divided by their maximum value and normalized as the evaluation of the classification difficulty. Finally, Flexmatch multiplys the fixed threshold by the classification difficulty metric of each category to get the dynamic thresholds for each category in current batch of unlabeled data. FlexMatch better alleviates the problem of class imbalance caused by different learning difficulties after unlabeled data is selected according to the confidence and does not cause excessive extra computing time and storage.

#### Deep Generative Model

Generative methods use real data to model a data distribution, and this distribution can be used to generate new data. Unlike classical generative models, deep generative models generate data based on deep neural networks. Generative Adversarial Network(GAN) and Variational Autoencoder(VAE) are the most commonly used generative models.

##### ImprovedGAN

Generative Adversarial Network is divided into two parts: the generator and the discriminator, where the generator assumes that the data can be generated by low-dimensional latent variables generated from a specific distribution and is used to generate simulated data by randomly sampling from the latent variable distribution. The generator is a classifier, which is used to distinguish whether the input sample is real data or simulated data generated by the generator. The generator is optimized to make the generated samples as close as possible to the real samples to deceive the discriminator and the discriminator is optimized to distinguish real data or simulated data as accurately as possible to avoid being deceived by the generator. The two are trained together in an adversarial manner, so as to achieve the purpose of obtaining a better generator and discriminator at the same time.

ImprovedGAN was proposed by Salimans et al. Classical GAN model can be trained only with unlabeled data, and its discriminator only needs to judge whether the sample is a real sample or a generated sample. ImprovedGAN adds the use of labeled data, requiring the discriminator not only to distinguish the authenticity of the samples, but also to complete the classification of the samples. So the discriminator is changed to a k+1 class classifier, where k is the number of classes in the original dataset. Both data generation and classification can be achieved through the alternate training of the generator and the discriminator,.

##### SSVAE

Variational Autoencoder integrates the deep autoencoder into the generative model. It also assumes that there are low-dimensional latent variables generated from a specific distribution. The latent variable is used as the representation vector of the original feature, and establish the mapping of latent variables to the original features as the decoder through the deep neural network. Since the posterior probability of meta-features to latent variables cannot be directly obtained, it also needs to be approximated by a neural network. As an encoder, the learning goal is to maximize the marginal probability of the original sample. VAT can learn a distribution that approximates the true posterior distribution and using it as an encoder can get a reasonable sample representation because when the approximate posterior distribution is equal to the true posterior distribution, the marginal probability can reach its upper bound.

SSVAE was proposed by Kingma et al. Classical VAE model can be trained only with unlabeled data, and its goal is to complete the learning of data representation through the encoder, and realize data generation through the decoder. SSVAE adds the application of labeled samples, and divides the encoder into two parts. The first part encodes the original data to obtain the probability distribution of the soft labels of the samples, and the second part uses the raw data and soft labels as input to obtain probability distributions of the hidden variables. The encoder of the classical VAE model only learns the representation of the data. The encoder of the SSVAE can firstly classify the samples and then can combine the sample information and the category information to learn the representation of the samples.

#### Deep Graph Based Method

When the raw data is a graph, since the instances are not independent but connected by edges, the classical deep learning method cannot effectively utilize the structural information of the graph model, so it cannot be directly applied to the graph data. However, graph data is very common in practical applications and it is of great significance to study deep learning methods that can be used for graph data. At present, graph deep learning has achieved certain research results. The classical semi-supervised learning methods ignores the structural information of the graph, so the effect of directly applying them to graph-structured data is not ideal. In reality, graph data tasks are often semi-supervised, because the nodes to be predicted and the training nodes are on the same graph. There are both labeled data and unlabeled data in the graph.

##### SDNE

SDNE was proposed by Wang et al. SDNE is a deep graph based semi-supervised learning method that can learn the embedding vector of nodes in the graph when there is no feature representation for the nodes in the graph and only graph structure information. This method adopts an autoencoder structure, takes the corresponding row of the node in the adjacency matrix as the adjacency vector of the node and inputs the adjacency vector of the node as the feature of the node into the self-encoder. SDNE obtains the embedded representation of the node through the encoder and restores the adjacency vector through the decoder. The loss function of SDNE mainly includes three items. The first item penalizes the inconsistency between the input and output of the autoencoder. In addition, unlike the classical autoencoder, the input of SDNE is an adjacency vector. Due to the sparseness of the adjacency matrix, there are a large number of zero values ​​in the input features. SDNE points out that more attention should be paid to the restoration of non-zero values, so zero and non-zero values ​​are given different weights. The second item is the Laplacian regularization which punishes the inconsistency of the hidden layer representation between adjacent nodes based on the graph structure information. The adjacency matrix is ​​used as the weight to obtain the Laplace regularization term. The third term is the L2 regularization, which penalizes the parameter complexity of the self-encoder to avoid overfitting. In SDNE, the first term of the loss function pays more attention to the characteristics of the node itself, while the second term pays more attention to the information between adjacent nodes which effectively solves the problem that the classical semi-supervised learning algorithm cannot effectively utilize the structural information of graphs.

##### GCN

GCN was proposed by Kipf et al. Unlike SDNE, which uses the adjacency vector of the node as the node feature to learn the embedding representation, GCN is more suitable for the situation that the node itself has features. In GCN, both the feature information and graph structure information of the node itself are available, which significantly improves the performance of the model. In graph deep learning, graph neural network(GNN) is the most commonly used class of methods. These methods usually take graphs with node features as input and can learn the deep representation of nodes to complete the learning assignment. The classical GNN method is divided into two steps: the first step is aggregation in which the information of the adjacent nodes are aggregated through the graph structure; the second step is update in which the nodes' representations are updated with their own representations and the information of their neighbors. By repeating these two steps, the deep representations of each node can be obtained. Due to the propagation effect of the aggregation operation, the deep representation of the node not only contains the information of the node itself, but also contains the information of the graph structure. The classical aggregation operation is linear aggregation which takes the linear combination of the representations of the neighbor nodes as the neighbor representation of the node. The classical update operation is to use the perceptron model to obtain new node representations from the nodes' own representations and their neighbor representations. The classical GNN model has some limitations. For examples, the aggregation method which linearly combines the representations of neighbor nodes makes nodes with larger numbers of degree have more influence while nodes with smaller numbers of degree have less influence on the entire training process. The GCN method directly adds the normalized neighbor representation to its own representation for each node and uses the result as the input of the perceptron to get a new representation. For each node, the normalization process divides the representation of its neighbor nodes and itself by the normalization factor, where the normalization factor of its neighbor nodes is the geometric mean of the degree of itself and its neighbor nodes while the normalization factor of itself is its own degree. GCN has excellent performance on graph structure tasks and its update process avoids the learning of linear combination weights of neighboring nodes so it has fewer parameters and higher efficiency.

# API

## Semi_sklearn.Algorithm

### Semi_sklearn.Algorithm.Classifiar

#### Semi_sklearn.Algorithm.Classifier.Assemble

> CLASS Semi_sklearn.Algorithm.Classifier.Assemble.Assemble(base_model=SVC(probability=True),T=100,alpha=1,beta=0.9)
>> Parameter
>> - base_model: A base learner for ensemble learning.
>> - T: the number of base learners. It is also the number of iterations.
>> - alpha: the weight of each sample when the sampling distribution is updated.
>> - Beta: used to initialize the sampling distribution of labeled data and unlabeled data.

#### Semi_sklearn.Algorithm.Classifier.Co_training

> CLASS Semi_sklearn.Algorithm.Classifier.Co_training.Co_training(base_estimator, base_estimator_2=None, p=5, n=5, k=30, s=75)
>> Parameter
>> - base_estimator: the first learner for co-training.
>> - base_estimator_2: the second learner for co-training.
>> - p: In each round, each base learner selects at most p positive samples to assign pseudo-labels.
>> - n: In each round, each base learner selects at most n negative samples to assign pseudo-labels.
>> - k: iteration rounds.
>> - s: the size of the buffer pool in each iteration.

#### Semi_sklearn.Algorithm.Classifier.Fixmatch

> CLASS Semi_sklearn.Algorithm.Classifier.Fixmatch.Fixmatch(self,train_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 train_dataloader=None,
                 valid_dataloader=None,
                 test_dataloader=None,
                 augmentation=None,
                 network=None,
                 train_sampler=None,
                 train_batch_sampler=None,
                 valid_sampler=None,
                 valid_batch_sampler=None,
                 test_sampler=None,
                 test_batch_sampler=None,
                 labeled_dataset=None,
                 unlabeled_dataset=None,
                 labeled_dataloader=None,
                 unlabeled_dataloader=None,
                 labeled_sampler=None,
                 unlabeled_sampler=None,
                 labeled_batch_sampler=None,
                 unlabeled_batch_sampler=None,
                 parallel=None,
                 epoch=1,
                 num_it_epoch=None,
                 num_it_total=None,
                 eval_epoch=None,
                 eval_it=None,
                 optimizer=None,
                 weight_decay=5e-4,
                 ema_decay=0.999,
                 scheduler=None,
                 device='cpu',
                 evaluation=None,
                 mu=1.0,
                 parallel=None,
                 file=None,
                 threshold=0.95,
                 lambda_u=1.0,
                 T=0.5)
>> Parameter
>> - threshold: choose the confidence threshold for the sample.
>> - lambda_u: Weight of unsupervised loss.
>> - T: the sharpening temperature.

#### Semi_sklearn.Algorithm.Classifier.Flexmatch
> CLASS Semi_sklearn.Algorithm.Classifier.Flexmatch.Flexmatch(self,train_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 train_dataloader=None,
                 valid_dataloader=None,
                 test_dataloader=None,
                 augmentation=None,
                 network=None,
                 train_sampler=None,
                 train_batch_sampler=None,
                 valid_sampler=None,
                 valid_batch_sampler=None,
                 test_sampler=None,
                 test_batch_sampler=None,
                 labeled_dataset=None,
                 unlabeled_dataset=None,
                 labeled_dataloader=None,
                 unlabeled_dataloader=None,
                 labeled_sampler=None,
                 unlabeled_sampler=None,
                 labeled_batch_sampler=None,
                 unlabeled_batch_sampler=None,
                 epoch=1,
                 num_it_epoch=None,
                 num_it_total=None,
                 eval_epoch=None,
                 eval_it=None,
                 optimizer=None,
                 weight_decay=5e-4,
                 scheduler=None,
                 device='cpu',
                 evaluation=None,
                 threshold=None,
                 mu=1.0,
                 ema_decay=None,
                 parallel=None,
                 file=None,
                 lambda_u=None,
                 T=None,
                 num_classes=10,
                 thresh_warmup=None,
                 use_hard_labels=False,
                 use_DA=False,
                 p_target=None)
>> Parameter
>> - threshold: The confidence threshold for choosing samples.
>> - lambda_u: Weight of unsupervised loss.
>> - T: Sharpening temperature.
>> - num_classes: The number of classes for the classification task.
>> - thresh_warmup: Whether to use threshold warm-up mechanism.
>> - use_hard_labels: Whether to use hard labels in the consistency regularization.
>> - use_DA: Whether to perform distribution alignment for soft labels.
>> - p_target: p(y) based on the labeled examples seen during training

#### Semi_sklearn.Algorithm.Classifier.GCN
> CLASS Semi_sklearn.Algorithm.Classifier.GCN(
                 epoch=1,
                 eval_epoch=None,
                 network=None,
                 optimizer=None,
                 weight_decay=None,
                 scheduler=None,
                 parallel=None,
                 file=None,
                 device='cpu',
                 evaluation=None,
                 num_features=1433,
                 num_classes=7,
                 normalize=True)
>> Parameter
>> - num_features: Node feature dimension.
>> - num_classes: number of classes.
>> - normalize: whether to use symmetric normalization.

#### Semi_sklearn.Algorithm.Classifier.ICT
> CLASS Semi_sklearn.Algorithm.Classifier.ICT(train_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 train_dataloader=None,
                 valid_dataloader=None,
                 test_dataloader=None,
                 augmentation=None,
                 network=None,
                 train_sampler=None,
                 train_batch_sampler=None,
                 valid_sampler=None,
                 valid_batch_sampler=None,
                 test_sampler=None,
                 test_batch_sampler=None,
                 labeled_dataset=None,
                 unlabeled_dataset=None,
                 labeled_dataloader=None,
                 unlabeled_dataloader=None,
                 labeled_sampler=None,
                 unlabeled_sampler=None,
                 labeled_batch_sampler=None,
                 unlabeled_batch_sampler=None,
                 eval_epoch=None,
                 eval_it=None,
                 optimizer=None,
                 weight_decay=None,
                 scheduler=None,
                 device='cpu',
                 evaluation=None,
                 epoch=1,
                 num_it_epoch=None,
                 num_it_total=None,
                 ema_decay=None,
                 mu=None,
                 parallel=None,
                 file=None,
                 warmup=None,
                 lambda_u=None,
                 alpha=None)
>> Parameter
>> - warmup: Warm up ratio for unsupervised loss.
>> - lambda_u: weight of unsupervised loss.
>> - alpha: the parameter of Beta distribution in Mixup.

## Base

### Semi_sklearn.SemiDeepModelMixin.SemiDeepModelMixin

> CLASS Semi_sklearn.Base.SemiDeepModelMixin.SemiDeepModelMixin(self, train_dataset=None,
                 labeled_dataset=None,
                 unlabeled_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 train_dataloader=None,
                 labeled_dataloader=None,
                 unlabeled_dataloader=None,
                 valid_dataloader=None,
                 test_dataloader=None,
                 augmentation=None,
                 network=None,
                 epoch=1,
                 num_it_epoch=None,
                 num_it_total=None,
                 eval_epoch=None,
                 eval_it=None,
                 mu=None,
                 optimizer=None,
                 weight_decay=5e-4,
                 ema_decay=None,
                 scheduler=None,
                 device=None,
                 evaluation=None,
                 train_sampler=None,
                 labeled_sampler=None,
                 unlabeled_sampler=None,
                 train_batch_sampler=None,
                 labeled_batch_sampler=None,
                 unlabeled_batch_sampler=None,
                 valid_sampler=None,
                 valid_batch_sampler=None,
                 test_sampler=None,
                 test_batch_sampler=None,
                 parallel=None,
                 file=None)
>> Parameter
>> - train_dataset
>> - labeled_dataset
>> - unlabeled_dataset
>> - valid_dataset
>> - test_dataset
>> - augmentation
>> - network
>> - epoch
>> - num_it_epoch
>> - num_it_total
>> - eval_epoch
>> - eval_it
>> - mu
>> - optimizer
>> - weight_decay
>> - ema_decay
>> - scheduler
>> - device
>> - evaluation
>> - train_sampler
>> - labeled_batch_sampler
>> - unlabeled_batch_sampler
>> - valid_sampler
>> - valid_batch_sampler
>> - test_sampler
>> - test_batch_sampler
>> - parallel
>> - file

## Dataloader

### Semi_sklearn.DataLoader.LabeledDataLoader.LabeledDataLoader
> CLASS Semi_sklearn.DataLoader.LabeledDataLoader.LabeledDataLoader(batch_size= 1, shuffle: bool = False,
                 sampler = None, batch_sampler= None,
                 num_workers: int = 0, collate_fn= None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn = None,
                 multiprocessing_context=None, generator=None,
                 prefetch_factor: int = 2, persistent_workers: bool = False)
>> Parameter
>> - batch_size
>> - shuffle
>> - sampler
>> - batch_sampler
>> - num_workers
>> - collate_fn
>> - pin_memory
>> - drop_last
>> - timeout
>> - worker_init_fn
>> - multiprocessing_context
>> - generator
>> - prefetch_factor
>> - persistent_workers

## Dataset
### Semi_sklearn.Dataset.LabeledDataset.LabeledDataset

> CLASS Semi_sklearn.Dataset.LabeledDataset.LabeledDataset（transforms=None, transform=None, target_transform=None, pre_transform=None)
>> Parameter
>> - transforms
>> - transform
>> - target_transform
>> - pre_transform

## Distributed
### Semi_sklearn.Distributed.DataParallel.DataParallel
> CLASS Semi_sklearn.DataParallel.DataParallel(device_ids=None, output_device=None, dim=0)
>> Parameter
>> - device_ids
>> - output_device
>> - dim

## Evaluation
### Semi_sklearn.Evaluation.Classification
#### Semi_sklearn.Evaluation.Classification.Accuracy
> CLASS Semi_sklearn.Evaluation.Classification.Accuracy(normalize=True, sample_weight=None)
>> Parameter
>> - normalize
>> - sample_weight

### Semi_sklearn.Evaluation.Regression
#### Semi_sklearn.Evaluation.Regression.Mean_absolute_error
> CLASS Semi_sklearn.Evaluation.Regression.Mean_absolute_error(sample_weight=None, multioutput="uniform_average")
>> Parameter
>> - sample_weight
>> - multioutput

## Loss
### Semi_sklearn.LOSS.Consistency
> CLASS Semi_sklearn.LOSS.Consistency(reduction='mean',activation_1=None,activation_2=None)
>> Parameter
>> - reduction
>> - activation_1
>> - activation_2

## Network
### Semi_sklearn.Network.GCN
> CLASS Semi_sklearn.Network.GCN(num_features,num_classes,normalize=False)
>> Parameter
>> - num_features
>> - num_classes
>> - normalize

## Optimizer
### Semi_sklearn.Optimizer.Adam
> CLASS Semi_sklearn.Optimizer.Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
>> Parameter
>> - lr
>> - betas
>> - eps
>> - weight_decay
>> - amsgrad

## Sampler
### Semi_sklearn.Sampler.RandomSampler
> CLASS Semi_sklearn.Sampler.RandomSampler(replacement: bool = False, num_samples = None, generator=None)
>> Parameter
>> - replacement
>> - num_samples
>> - generator

## Scheduler
### Semi_sklearn.Scheduler.CosineAnnealingLR
> CLASS Semi_sklearn.Scheduler.CosineAnnealingLR(T_max, eta_min=0, last_epoch=-1, verbose=False)
>> Parameter
>> - T_max
>> - eta_min
>> - last_epoch
>> - verbose

## Split
### Semi_sklearn.Scheduler.Split.SemiSplit
> Function Semi_sklearn.Scheduler.Split.SemiSplit(stratified, shuffle, random_state=None, X=None, y=None,labeled_size=None)
>> Parameter
>> - stratified
>> - shuffle
>> - random_state
>> - X
>> - y
>> - labeled_size

## Transform
### Semi_sklearn.Transform.Adjust_length
> CLASS Semi_sklearn.Transform.Adjust_length(length, pad_val=None, pos=0)
>> Parameter
>> - length
>> - pad_val
>> - pos

## utils

### Semi_sklearn.utils.get_indexing_method
> Function Semi_sklearn.utils.get_indexing_method(data)
>> Parameter
>> - data

# FAQ
1. What is the difference of interfaces between Semi-sklearn and the semi-supervised learning module of sklearn?

The fit() method of sklearn generally has two items, X and y. The label y corresponding to the unlabeled X is represented by -1. But in many binary classification tasks, -1 represents a negative class, which is easy to conflict. So the fit() method of Semi-sklearn has three inputs of X, y and unlabeled_X.

2. How to understand the DeepModelMixin module?

This module mainly makes deep learning and classical machine learning have the same interface. And in order to facilitate users to replace the corresponding components of deep learning, DeepModelMixin decouples components of pytorch.

# Reference

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