<div align=center>
<img width="1000px" src="./Imgs/Logo1.png"> 

![](https://badgen.net/pypi/v/LAMDA-SSL)
![](https://anaconda.org/ygzwqzd/lamda-ssl/badges/version.svg)
![](https://badgen.net/github/stars/YGZWQZD/LAMDA-SSL)
![visitors](https://visitor-badge.glitch.me/badge?page_id=ygzwqzd.LAMDA-SSL)
![](https://badgen.net/github/last-commit/YGZWQZD/LAMDA-SSL)
![](https://badgen.net/github/license/YGZWQZD/LAMDA-SSL)

**[Documentation](https://ygzwqzd.github.io/LAMDA-SSL)** | **[Paper]()** |  **[Examples](https://github.com/ygzwqzd/LAMDA-SSL/tree/master/Examples)**

</div>

#  Introduction

In order to promote the research and application of semi-supervised learning algorithms, we has developed LAMDA which is a convenient and practical semi-supervised learning toolkit. LAMDA-SSL has complete functions, convenient interfaces and detailed documentations. It integrates statistical machine learning algorithms and deep learning algorithms into the same framework. It is compatible with the popular machine learning toolkit sklearn and the popular deep learning toolkit pytorch.  It supports Pipeline mechanism and parameter search functions of sklearn and also supports GPU acceleration and distributed training functions of pytorch. At present, LAMDA-SSL contains 30 semi-supervised learning algorithms, including 12 algorithms based on statistical machine learning models and 18 algorithms based on deep learning models. LAMDA-SSL also contains 45 data processing methods used for 4 types of data: table, image, text, graph and 15 model evaluation criterias used for 3 types of task: classification, regression and clustering. LAMDA-SSL includes multiple modules such as data management, data transformation, algorithm application, model deployment and so on, which facilitates the implementation of end to end semi-supervised learning.

<div align=center>
<img width="1000px" src="./Imgs/Overview.png"> 
</div>

At present, LAMDA-SSL has implemented 30 semi-supervised learning algorithms, including 12 statistical semi-supervised learning algorithms and 18 deep semi-supervised learning algorithms. 

For statistical semi-supervised learning, algorithms in LAMDA-SSL can be used for classification, regression and clustering. The algorithms used for classification task include generative method SSGMM; semi-supervised support vector machine methods TSVM and LapSVM; graph-based methods Label Propagation and Label Spreading; disagrement-based methods Co-Training and Tri-Training; ensemble methods SemiBoost and Assemble. The algorithm used for regression task is CoReg. The algorithms used for clustering task include Constrained K Means, Constrained Seed K Means.
<div align=center>
<img width="1000px" src="./Imgs/Statistical.png"> 
</div>
For deep semi-supervised learning (as shown in Figure 3), algorithms in LAMDA-SSL can be used for classification and regression. The algorithms used for classification task include consistency methods Ladder Network, Π Model, Temporal Ensembling, Mean Teacher, VAT and UDA; Pseudo label-based methods Pseudo Label and S4L; hybrid methods ICT, MixMatch, ReMixMatch, FixMatch and FlexMatch; deep generative methods ImprovedGAN and SSVAE; deep graph-based methods SDNE and GCN. The algorithms for regression task include consistency method Π Model Reg, Mean Teacher Reg; hybrid method ICT Reg. These 3 deep SSL regression algorithms are our extensions of their prototypes used for classification.
<div align=center>
<img width="1000px" src="./Imgs/Deep.png"> 
</div>


# Advantages

> - LAMDA-SSL contains 30 semi-supervised learning algorithms.
> - LAMDA-SSL can handle 4 types of data and has rich data processing functions.
> - LAMDA-SSL can handle 3 types of tasks and has rich metrics for model evaluation.
> - LAMDA-SSL supports both statistical semi-supervised learning algorithms and deep semi-supervised learning algorithms.
> - LAMDA-SSL is compatible with the popular machine learning toolkit sklearn and the popular deep learning toolkit pytorch.
> - LAMDA-SSL has simple interfaces similiar to sklearn so that it is easy to use.
> - LAMDA-SSL has powerful functions. It supports Pipeline mechanism and parameter search functions of sklearn and also supports GPU acceleration and distributed training functions of pytorch.
> - LAMDA-SSL considers the needs of different user groups. It provides well tuned default parameters and modules for low-proficiency users. It also supports flexible module replacement for high-proficiency users.
> - LAMDA-SSL has strong extensibility, which is convenient for users to customize new modules and algorithms.
> - LAMDA-SSL has been verified by a large number of experiments and has strong reliability.
> - LAMDA-SSL has complete user documentations.


# Dependencies

LAMDA-SSL requires:

> - python (>= 3.7)
> - scikit-learn (>= 1.0.2)
> - torch (>= 1.9.0)
> - torchvision (>= 0.10.0)
> - torchtext (>= 0.10.0)
> - torch-geometric(>= 2.0.3)
> - Pillow(>= 8.4.0)
> - numpy(>= 1.21.5)
> - scipy(>= 1.7.3)
> - pandas(>= 1.3.4)
> - matplotlib(>= 3.5.0)

# Installation

## Install from pip

You can download LAMDA-SSL directly from pip.
```
pip install LAMDA-SSL
```

## Install from anaconda

You can also download LAMDA-SSL directly from anaconda.
```
conda install -c ygzwqzd LAMDA-SSL
```


## Install from the source

If you want to try the latest features that have not been released yet, you can install LAMDA-SSL from the source.
```
git clone https://github.com/ygzwqzd/LAMDA-SSL.git
cd LAMDA-SSL
pip install .
```

# Quick Start

For example, train a FixMatch classifier for CIFAR10.

Firstly, import and initialize CIFAR10.

```python
from LAMDA_SSL.Dataset.Vision.CIFAR10 import CIFAR10

dataset = CIFAR10(root='..\Download\cifar-10-python',
                  labeled_size=4000, stratified=False,
                  shuffle=True, download=True)
labeled_X = dataset.labeled_X
labeled_y = dataset.labeled_y
unlabeled_X = dataset.unlabeled_X
test_X = dataset.test_X
test_y = dataset.test_y
```

Then import and initialize FixMatch.

```python
from LAMDA_SSL.Algorithm.Classification.FixMatch import FixMatch
model=FixMatch(threshold=0.95,lambda_u=1.0,mu=7,T=0.5,epoch=1,num_it_epoch=2**20,device='cuda:0')
```

Next, call the fit() method to complete the training process of the model.
```python
model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
```

Finally, call the predict() method to predict the labels of new samples.
```python
pred_y=model.predict(X=test_X)
```

# Performance

We have evaluated the performance of LAMDA-SSL for semi-supervised classification task on table data using BreastCancer dataset. In this experiment, 30% of the instances are randomly sampled to form the testing dataset by the class distribution. Then 10% of the remaining instances are randomly sampled to form the labeled training dataset and the others are used to form the unlabeled training dataset by dropping their labels. For detailed parameter settings of each method, please refer to the 'Config' module of LAMDA-SSL.

<div align=center>

| Method | Accuracy(%) | F1 Score |
| :-: | :-: | :-: |
| SSGMM | 94.74 | 94.43 |
| TSVM | 92.40 | 91.62 |
| LapSVM | 96.49| 96.20 |
| Label Propagation| 93.57| 92.85 |
| Label Spreading | 95.32 | 94.90 |
| Co-Training| 97.08| 99.07 |
| Tri-Training| 97.66| 97.47 |
| Assemble | 94.15| 93.75 |
| SemiBoost | 97.08 | 96.85 |

</div>

We have evaluated the performance of LAMDA-SSL for semi-supervised regression task on table data using Boston dataset. In this experiment, 30% of the instances are randomly sampled to form the testing dataset by the class distribution. Then 10% of the remaining instances are randomly sampled to form the labeled training dataset and the others are used to form the unlabeled training dataset by dropping their labels. For detailed parameter settings of each method, please refer to the 'Config' module of LAMDA-SSL.

<div align=center>

| Method | Mean Absolute Error | Mean Squared Error |
| :-: | :-: | :-: |
|CoReg|	4.66|	59.52|
|Π Model Reg|	4.32|	37.64|
|ICT Reg	|4.11|	37.14|
|Mean Teacher Reg	|4.51|	45.56|

</div>

We have evaluated the performance of LAMDA-SSL for semi-supervised clustring task on table data using Wine dataset. In this experiment, 20% of the instances are randomly sampled to form the labeled dataset and the others are used to form the unlabeled dataset by dropping their labels. For detailed parameter settings of each method, please refer to the 'Config' module of LAMDA-SSL.

<div align=center>

| Method | Davies Bouldin Score | Fowlkes Mallows Score |
| :-: | :-: | :-: |
|Constrained k-means	|1.76	|0.75|
|Constrained Seed k-means	|1.38|	0.93|

</div>

We have evaluated the performance of LAMDA-SSL for semi-supervised clustring task on simple vision data using MNIST dataset. In this experiment, 10% of the instances in training dataset are randomly sampled to form the labeled dataset and the others are used to form the unlabeled dataset by dropping their labels. For detailed parameter settings of each method, please refer to the 'Config' module of LAMDA-SSL.

<div align=center>

| Method | Accuracy(%) | F1 Score |
| :-: | :-: | :-: |
|Ladder Network	|97.37	|97.36|
|ImprovedGAN	|98.81|	98.81|
|SSVAE|	96.69|	96.67|

</div>

We have evaluated the performance of LAMDA-SSL for semi-supervised classification task on complex vision data using CIFAR10 dataset. In this experiment, 4000 instances in training dataset are randomly sampled to form the labeled training dataset and the others are used to form the unlabeled training dataset by dropping their labels. WideResNet is used as the backbone network. For detailed parameter settings of each method, please refer to the 'Config' module of LAMDA-SSL.

<div align=center>

| Method | Accuracy(%) | F1 Score |
| :-: | :-: | :-: |
|UDA	|95.41|	95.40|
|Π Model	|87.09|	87.07|
|Temporal Ensembling|	89.30|	89.31|
|Mean Teacher|	92.01	|91.99|
|VAT	|88.22	|88.19|
|Pseudo Label|	85.90|	85.85|
|S4L	|89.59	|89.54|
|ICT	|92.64	|92.62|
|MixMatch	|93.43	|93.43|
|ReMixMatch	|96.24	|96.24|
|FixMatch	|95.34	|95.33|
|FlexMatch|	95.39	|95.40|


</div>

We have evaluated the performance of LAMDA-SSL for semi-supervised classification task on graph data using Cora dataset. In this experiment, 20% of the instances are randomly sampled to form the labeled training dataset and the others are used to form the unlabeled training dataset by dropping their labels. For detailed parameter settings of each method, please refer to the 'Config' module of LAMDA-SSL.

<div align=center>

| Method | Accuracy(%) | F1 Score |
| :-: | :-: | :-: |
|SDNE|	73.78|	69.85|
|GCN|	82.04|	80.52|
|GAT|	79.13|	77.36|


</div>

# Citation
Please cite our paper if you find LAMDA-SSL useful in your work:
```

```

# Contribution
Feel free to contribute in any way you like, we're always open to new ideas and approaches.
- [Open a discussion](https://github.com/YGZWQZD/LAMDA-SSL/discussions/new) if you have any question.
- Feel welcome to [open an issue](https://github.com/YGZWQZD/LAMDA-SSL/issues/new) if you've spotted a bug or a performance issue.
- Learn more about how to customize modules of LAMDA-SSL from the [Usage](https://ygzwqzd.github.io/LAMDA-SSL/#/README?id=usage) section of the [documentation](https://ygzwqzd.github.io/LAMDA-SSL/#/).

# The Team
 LAMDA-SSL is developed by [LAMDA](https://www.lamda.nju.edu.cn/MainPage.ashx)12@[NJU](https://www.nju.edu.cn/en/main.psp). Contributors are .

# Contact
If you have any questions, please contact us: Lin-Han Jia[jialh2118@gmail.com].
