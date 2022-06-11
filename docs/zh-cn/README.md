#  介绍

Semi-sklearn是一个有效易用的半监督学习工具包。目前该工具包包含30种半监督学习算法，其中基于传统机器学习模型的算法13种，基于深度神经网络模型的算法17种，可用于处理结构化数据、图像数据、文本数据、图结构数据4种数据类型，可用于分类、回归、聚类3种任务，包含数据管理、数据变换、算法应用、模型评估等多个模块，便于实现端到端的半监督学习过程，兼容目前主流的机器学习工具包scikit-learn和深度学习工具包pytorch，具备完善的功能，标准的接口和详尽的文档。


##  设计思想

Semi-sklearn的整体设计思想如图所示。Semi-sklearn参考了sklearn工具包的底层实现，所有算法都使用了与sklearn相似的接口。 在sklearn中的学习器都继承了Estimator这一父类，Estimator表示一个估计器，利用现有数据建立模型对未来的数据做出预测，对估计器存在fit()和transform()两个方法，其中fit()方法是一个适配过程，即利用现有数据建立模型，对应了机器学习中的训练过程，transform()方法是一个转换过程，即利用fit()过后的模型对新数据进行预测。

Semi-sklearn中的预测器通过继承半监督预测器类SemiEstimator间接继承了sklearn中的Estimator。由于sklearn中fit()方法使用的数据往往包含样本和标注两项，在半监督学习中，模型的训练过程中同时使用有标注数据、标注和无标注数据，因此Estimator的fit()方法不方便直接用于半监督学习算法。虽然sklearn中也实现了自训练方法和基于图的方法两类半监督学习算法，它们也继承了Estimator类，但是为了使用fit()方法的接口，sklearn将有标注样本与无标注数据样本结合在一起作为fit()的样本输入，将标注输入中无标注数据对应的标注记为-1，这种处理方式虽然可以适应Estimator的接口，但是也存在局限性，尤其使在一些二分类场景下往往用-1表示有标注数据的负例标注，与无标注数据会发生冲突，因此针对半监督学习在Estimator的基础上重新建立新类SemiEstimator具有必要性，SemiEstimator的fit()方法包含有标注数据、标注和无标注数据三部分输入，更好地契合了半监督学习的应用场景，避免了要求用户自己对数据进行组合处理，也避免了无标注数据与二分类负类的冲突，相较Estimator使用起来更加方便。

半监督学习一般分为归纳式学习和直推式学习，区别在于是否直接使用待预测数据作为训练过程中的无标注数据。Semi-sklearn中使用两个类InductiveEstimator和Transductive分别对应了归纳式学习和直推式学习两类半监督学习方法，均继承了SemiEstimator类。

在sklearn中，为了使估计器针对不同的任务可以具备相应的功能，sklearn针对估计器的不同使用场景开发了与场景对应的组件（Mixin），sklearn中的估计器往往会同时继承Estimator和相应组件，从而使估计器同时拥有基本的适配和预测功能，还能拥有不同组件对应的处理不同任务场景的功能。其中关键组件包括用于分类任务的ClassifierMixin、用于回归任务的RegressorMixin、用于聚类任务的ClusterMixin和用于数据转换的TransformerMixin，在Semi-sklearn中同样使用了这些组件。

另外，不同于经典机器学习中常用的sklearn框架，深度学习在经常使用pytorch框架，pytorch各组件间存在较大的依赖关系（如图3-2所示），耦合度高，例如数据集（Dataset）与数据加载器（Dataloader）的耦合、优化器（Optimizer）和调度器（Scheduler）的耦合、采样器（Sampler）与批采样器（BatchSampler）的耦合等，没有像sklearn一样的简单的逻辑和接口，对用户自身要求较高，较不方便，为在同一工具包在同时包含经典机器学习方法和深度学习方法造成了较大困难，为了解决经典机器学习方法和深度学习方法难以融合于相同框架的问题，Semi-sklearn用DeepModelMixin这一组件使基于pytorch开发的深度半监督模型拥有了与经典机器学习方法相同接口和使用方式，Semi-sklearn中的深度半监督学习算法都继承了这一组件。DeepModelMixin对pytorch各模块进行了解耦，便于用户独立更换深度学习中数据加载器、网络结构、优化器等模块，而不需要考虑更换对其他模块造成的影响，DeepModelMixin会自动处理这些影响，使用户可以像调用经典的半监督学习算法一样便捷地调用深度半监督学习算法。

#  Data Management

Semi-sklearn拥有强大的数据管理和数据处理功能。在Semi-sklearn中，一个半监督数据集整体可以用一个SemiDataset类进行管理，SemiDataset类可以同时管理TrainDataset、ValidDataset、TestDataset三个子数据集，分别对应了机器学习任务中的训练数据集、验证数据集和测试数据集，在最底层数据集分为LabeledDataset和UnlabeledDataset两类，分别对应了半监督学习中的有标注数据与无标注数据，训练集往往同时包含有标注数据和无标注数据，因此TrainDataset同时管理LabeledDataset和UnlabeledDataset两个数据集。

Semi-sklearn针对LabeledDataset和UnlabeledDataset分别设计了LabeledDataloader和UnlabeledDataloader两种数据加载器，而用一个TrainDataloader类同时管理两种加载器用于半监督学习的训练过程，除同时包含两个加载器外，还起到调节两个加载器之间关系的作用，如调节每一批数据中有标注数据与无标注数据的比例。

Semi-sklearn可以处理结构化数据、图像数据、文本数据、图数据四种现实应用中常见的数据类型，分别使用了四个与数据类型对应的组件StructuredDataMixin、VisionMixin、TextMixin、GraphMixin进行处理，对于一个数据集，可以继承与其数据类型对应的组件获得组件中的数据处理功能。

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

# FAQ
1. Semi-sklearn的接口和sklearn半监督学习模块的接口有什么不同？
sklearn的接口的fit()方法一般有X和y两项，无标注的X对应的标注y用-1表示。但是在很多二分类任务中，-1表示负类，容易冲突，因此Semi-sklearn的fit()方法有X,y和unlabeled_X三项输入。
2. DeepModelMixin模块如何理解？
这一模块主要是使深度学习与经典机器学习拥有相同的接口，并且为了便于用户更换深度学习种对应的组件，DeepModelMixin对pytorch进行了解耦。










