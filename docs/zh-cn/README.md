#  介绍

Semi-sklearn是一个有效易用的半监督学习工具包。目前该工具包包含30种半监督学习算法，其中基于传统机器学习模型的算法13种，基于深度神经网络模型的算法17种，可用于处理结构化数据、图像数据、文本数据、图结构数据4种数据类型，可用于分类、回归、聚类3种任务，包含数据管理、数据变换、算法应用、模型评估等多个模块，便于实现端到端的半监督学习过程，兼容目前主流的机器学习工具包scikit-learn和深度学习工具包pytorch，具备完善的功能，标准的接口和详尽的文档。


#  设计思想

Semi-sklearn的整体设计思想如图所示。Semi-sklearn参考了sklearn工具包的底层实现，所有算法都使用了与sklearn相似的接口。 在sklearn中的学习器都继承了Estimator这一父类，Estimator表示一个估计器，利用现有数据建立模型对未来的数据做出预测，对估计器存在fit()和transform()两个方法，其中fit()方法是一个适配过程，即利用现有数据建立模型，对应了机器学习中的训练过程，transform()方法是一个转换过程，即利用fit()过后的模型对新数据进行预测。

Semi-sklearn中的预测器通过继承半监督预测器类SemiEstimator间接继承了sklearn中的Estimator。由于sklearn中fit()方法使用的数据往往包含样本和标注两项，在半监督学习中，模型的训练过程中同时使用有标注数据、标注和无标注数据，因此Estimator的fit()方法不方便直接用于半监督学习算法。虽然sklearn中也实现了自训练方法和基于图的方法两类半监督学习算法，它们也继承了Estimator类，但是为了使用fit()方法的接口，sklearn将有标注样本与无标注数据样本结合在一起作为fit()的样本输入，将标注输入中无标注数据对应的标注记为-1，这种处理方式虽然可以适应Estimator的接口，但是也存在局限性，尤其使在一些二分类场景下往往用-1表示有标注数据的负例标注，与无标注数据会发生冲突，因此针对半监督学习在Estimator的基础上重新建立新类SemiEstimator具有必要性，SemiEstimator的fit()方法包含有标注数据、标注和无标注数据三部分输入，更好地契合了半监督学习的应用场景，避免了要求用户自己对数据进行组合处理，也避免了无标注数据与二分类负类的冲突，相较Estimator使用起来更加方便。



#  Data Management

![Dataset](./Imgs/Dataset.png)

#  Model



##  Classical  Method



###  Semi Supervised SVM



####  S3VM(TSVM) (√)



#### LapSVM(√)



###  Graphed Based Method



####  Label Propagation (√)



####  Label Spreading (√)



### Generative Method



####  Semi Supervised Gaussian Mixture Model (√)



###  Wrapper Method 



#### Self Training (√)



####  Co Training (√)



####  Tri Training (√)



###  Semi Supervised Cluster



#### Constrained K Means (√)



#### Constrained Seed K Means (√)



###  Semi Supervised Regression



####  CoReg (√)



###  Ensemble Learning



####  Semi Boost (√)



####  Assemble AdaBoost (√)



## Deep Learning Method



###  Consistency Regularization



####  Pi Model (√)



####  Temporal Ensembling (√)



#### Mean Teacher (√)



####  VAT (√)



####  UDA (√)



####  Ladder Network (√)



###  Pseudo Labeling



####  Pseudo Label (√)



####  S4L (√)



###  Generative Method



####  ImprovedGan (√)



####  SSVAE (√)



### Graph-based methods



####  SDNE (√)



####  GCN (√)



###  Hybrid Method



####  FixMatch (√)



####  MixMatch (√)



####  ReMixMatch (√)



#### FlexMatch (√)



####  ICT (√)

















