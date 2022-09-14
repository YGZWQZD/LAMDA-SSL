from LAMDA_SSL.Algorithm.Regression.CoReg import CoReg
from LAMDA_SSL.Algorithm.Regression.ICTReg import ICTReg
from LAMDA_SSL.Algorithm.Regression.PiModelReg import PiModelReg
from LAMDA_SSL.Algorithm.Regression.MeanTeacherReg import MeanTeacherReg
from LAMDA_SSL.utils import class_status
from sklearn.preprocessing import QuantileTransformer,OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import ColumnTransformer
from KMM import KMM
from LAMDA_SSL.Algorithm.Classification.TSVM import TSVM
from Self_Training import Self_Training
from LAMDA_SSL.Dataset.Vision.Mnist import Mnist
from Pseudo_Label import Pseudo_Label
from LAMDA_SSL.Split.DataSplit import DataSplit
import copy
from sklearn.linear_model._logistic import LogisticRegression
import pickle
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Evaluation.Classifier.Precision import Precision
from LAMDA_SSL.Evaluation.Classifier.Recall import Recall
from LAMDA_SSL.Evaluation.Classifier.F1 import F1
from LAMDA_SSL.Evaluation.Classifier.AUC import AUC
from LAMDA_SSL.Evaluation.Classifier.Confusion_Matrix import Confusion_Matrix
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import random
path='../../data/numerical_only/balanced/'
file = open("imbalanced_class.txt", "w")
label_distribution=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]

# dataset=Mnist(root='..\Download\mnist',labeled_size=6000,shuffle=True,download=False,random_state=0,default_transforms=True)
#
# labeled_X=dataset.labeled_X
# labeled_y=dataset.labeled_y
#
# unlabeled_X=dataset.unlabeled_X
#
# valid_X=dataset.valid_X
# valid_y=dataset.valid_y
#
# test_X=dataset.test_X
# test_y=dataset.test_y

# print(test_y)

# def distribution_selection(X,y,gamma=1e-5):
#     # print(X.shape)
#     # print(y.shape)
#     mean_X=np.mean(X,axis=0)
#     p=[]
#     s=[]
#     r=np.random.random(X.shape[0])
#     for _ in range(X.shape[0]):
#         p_s=np.exp(-gamma*np.linalg.norm(mean_X-X[_])**2)
#         p.append(p_s)
#         if r[_]<p_s:
#             s.append(1)
#         else:
#             s.append(0)
#     print(s.count(0))
#     print(s.count(1))
#     s=np.array(s)
#     s_X=X[s==1]
#     s_y=y[s==1]
#     r_s=random.sample(list(range(X.shape[0])),s_X.shape[0])
#     r_X=X[r_s]
#     r_y=y[r_s]
#     # print(s_X.shape)
#     # print(s_y.shape)
#     # print(r_X.shape)
#     # print(r_y.shape)
#     return s_X,s_y,r_X,r_y

def distribution_selection_2(X,y,p1=0.90,p2=0.10):
    mean_X=np.mean(X,axis=0)
    r=np.random.random(X.shape[0])
    s = []
    j = random.choice(list(range(X.shape[1])))
    for i in range(X.shape[0]):
        flag=False
        if X[i][j]<mean_X[j]:
            if r[i]<p1:
                flag=True
        else:
            if r[i]<p2:
                flag=True
        if flag:
            s.append(1)
        else:
            s.append(0)
    print(s.count(0))
    print(s.count(1))
    s=np.array(s)
    s_X=X[s==1]
    s_y=y[s==1]
    r_s=random.sample(list(range(X.shape[0])),s_X.shape[0])
    r_X=X[r_s]
    r_y=y[r_s]
    # print(s_X.shape)
    # print(s_y.shape)
    # print(r_X.shape)
    # print(r_y.shape)
    return s_X,s_y,r_X,r_y

# def distribution_selection_3(X,y,p1=0.90,p2=0.10):
#     mean_X=np.mean(X,axis=0)
#     r=np.random.random(X.shape[0])
#     s = []
#     j = random.choice(list(range(X.shape[1])))
#     for i in range(X.shape[0]):
#         flag=False
#         if X[i][j]<mean_X[j]:
#             if r[i]<p1:
#                 flag=True
#         else:
#             if r[i]<p2:
#                 flag=True
#         if flag:
#             s.append(1)
#         else:
#             s.append(0)
#     print(s.count(0))
#     print(s.count(1))
#     s=np.array(s)
#     s_X=X[s==1]
#     s_y=y[s==1]
#     r_s=random.sample(list(range(X.shape[0])),s_X.shape[0])
#     r_X=X[r_s]
#     r_y=y[r_s]
#     return s_X,s_y,r_X,r_y
def distribution_selection_4(X,y,p1=0.90,p2=0.10):
    mean_X=np.mean(X,axis=0)
    r=np.random.random(X.shape[0])
    s = []
    for i in range(X.shape[0]):
        flag=False
        if y[i]==1:
            if r[i]<p1:
                flag=True
        else:
            if r[i]<p2:
                flag=True
        if flag:
            s.append(1)
        else:
            s.append(0)
    print(s.count(0))
    print(s.count(1))
    s=np.array(s)
    s_X=X[s==1]
    s_y=y[s==1]
    r_s=random.sample(list(range(X.shape[0])),s_X.shape[0])
    r_X=X[r_s]
    r_y=y[r_s]
    # print(s_X.shape)
    # print(s_y.shape)
    # print(r_X.shape)
    # print(r_y.shape)
    return s_X,s_y,r_X,r_y
data_id =[
                722,
                821,
                993,
                1120,
                1461,
                1489,
                1044,
                None,
                None,
                None,
                151,
                41168,
41150,293,42769
               ]
dataset=[
        "pol",
        "house_16H",
        "kdd_ipums_la_97-small",
        "MagicTelescope",
        "bank-marketing",
         "phoneme",
        "eye_movements",
        "credit",
        "california",
        "wine",
        "electricity",
        "jannis",
        "MiniBooNE",
        "covertype",
        "Higgs"
    ]

# evalutation
evaluation= Accuracy()

for _ in range(len(dataset)):
    data_name=dataset[_]
    print(data_name)
    id=data_id[_]
    data_path= path+'data_'+data_name
    with open(data_path,"rb") as f:
        data = pickle.load(f)
    X,y=data
    # print(X.shape)
    # print(y.shape)

    l_performance1=[]
    l_performance2=[]
    l_performance3=[]
    l_performance4=[]

    for _ in range(5):
        test_X, test_y, train_X, train_y = DataSplit(stratified=True, shuffle=True, random_state=_, X=X, y=y,
                                                     size_split=0.2)
        test_X, test_y,rtx,rty=distribution_selection_2(test_X, test_y, p1=0.1, p2=0.9)
        trans = QuantileTransformer(output_distribution='normal').fit(train_X)
        test_X = trans.transform(test_X)
        _train_X=copy.deepcopy(train_X)
        _train_y=copy.deepcopy(train_y)
        labeled_X,labeled_y,unlabeled_X,unlabeled_y=DataSplit(stratified=True,shuffle=True,random_state=_, X=_train_X, y=_train_y,size_split=0.375)
        # s1_unlabeled_X, s1_unlabeled_y, r1_unlabeled_X, r1_unlabeled_y = distribution_selection(unlabeled_X, unlabeled_y)
        labeled_X, labeled_y, rlx, rly = distribution_selection_2(labeled_X, labeled_y,p1=0.1,p2=0.9)
        s2_unlabeled_X,s2_unlabeled_y,r2_unlabeled_X,r2_unlabeled_y=distribution_selection_2(unlabeled_X,unlabeled_y)
        # s3_unlabeled_X, s3_unlabeled_y, r3_unlabeled_X, r3_unlabeled_y = distribution_selection_3(unlabeled_X, unlabeled_y)
        # beta = KMM().fit(labeled_X, s1_unlabeled_X)
        # print(beta.shape)
        labeled_X=trans.transform(labeled_X)
        # s1_unlabeled_X=trans.transform(s1_unlabeled_X)
        # r1_unlabeled_X = trans.transform(r1_unlabeled_X)
        s2_unlabeled_X = trans.transform(s2_unlabeled_X)
        r2_unlabeled_X = trans.transform(r2_unlabeled_X)
        # s3_unlabeled_X = trans.transform(s2_unlabeled_X)
        # r3_unlabeled_X = trans.transform(r2_unlabeled_X)


        # print('kmm_pseudo')
        # self_training_model = Pseudo_Label(base_estimator=LogisticRegression(),evaluation=evaluation,kmm=True)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=s1_unlabeled_X)
        # performance5 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance5)
        #
        # self_training_model = Pseudo_Label(base_estimator=LogisticRegression(),evaluation=evaluation,kmm=False)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=s1_unlabeled_X)
        # performance1 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance1)
        #
        # self_training_model = Pseudo_Label(base_estimator=LogisticRegression(),evaluation=evaluation,kmm=False)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=r1_unlabeled_X)
        # performance2 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance2)

        print('kmm_pseudo')

        self_training_model = Pseudo_Label(base_estimator=GradientBoostingClassifier(),evaluation=evaluation,kmm=True)
        self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=s2_unlabeled_X)
        performance1 = self_training_model.evaluate(X=test_X, y=test_y)
        print(performance1)
        l_performance1.append(performance1)

        self_training_model = Pseudo_Label(base_estimator=GradientBoostingClassifier(),evaluation=evaluation,kmm=False)
        self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=s2_unlabeled_X)
        performance2 = self_training_model.evaluate(X=test_X, y=test_y)
        print(performance2)
        l_performance2.append(performance2)

        self_training_model = Pseudo_Label(base_estimator=GradientBoostingClassifier(),evaluation=evaluation,kmm=True)
        self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=r2_unlabeled_X)
        performance3 = self_training_model.evaluate(X=test_X, y=test_y)
        print(performance3)
        l_performance3.append(performance3)

        self_training_model = Pseudo_Label(base_estimator=GradientBoostingClassifier(),evaluation=evaluation,kmm=False)
        self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=r2_unlabeled_X)
        performance4 = self_training_model.evaluate(X=test_X, y=test_y)
        print(performance4)
        l_performance4.append(performance4)

        # print('kmm_pseudo')

        # self_training_model = Pseudo_Label(base_estimator=LogisticRegression(),evaluation=evaluation,kmm=True)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=s3_unlabeled_X)
        # performance6 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance6)
        #
        # self_training_model = Pseudo_Label(base_estimator=LogisticRegression(),evaluation=evaluation,kmm=False)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=s3_unlabeled_X)
        # performance3 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance3)
        #
        # self_training_model = Pseudo_Label(base_estimator=LogisticRegression(),evaluation=evaluation,kmm=False)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=r3_unlabeled_X)
        # performance4 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance4)

        # print('kmm_self')
        # self_training_model = Self_Training(base_estimator=LogisticRegression(),evaluation=evaluation,kmm=True)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=s1_unlabeled_X)
        # performance5 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance5)
        #
        # self_training_model = Self_Training(base_estimator=LogisticRegression(),evaluation=evaluation,kmm=False)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=s1_unlabeled_X)
        # performance1 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance1)
        #
        # self_training_model = Self_Training(base_estimator=LogisticRegression(),evaluation=evaluation,kmm=False)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=r1_unlabeled_X)
        # performance2 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance2)

        # print('kmm_self')
        #
        # self_training_model = Self_Training(base_estimator=GradientBoostingClassifier(),evaluation=evaluation,kmm=True)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=s2_unlabeled_X)
        # performance6 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance6)
        #
        # self_training_model = Self_Training(base_estimator=GradientBoostingClassifier(),evaluation=evaluation,kmm=False)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=s2_unlabeled_X)
        # performance3 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance3)
        #
        # self_training_model = Self_Training(base_estimator=GradientBoostingClassifier(),evaluation=evaluation,kmm=False)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=r2_unlabeled_X)
        # performance4 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance4)

        # print('kmm_self')
        #
        # self_training_model = Self_Training(base_estimator=LogisticRegression(),evaluation=evaluation,kmm=True)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=s3_unlabeled_X)
        # performance6 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance6)
        #
        # self_training_model = Self_Training(base_estimator=LogisticRegression(),evaluation=evaluation,kmm=False)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=s3_unlabeled_X)
        # performance3 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance3)
        #
        # self_training_model = Self_Training(base_estimator=LogisticRegression(),evaluation=evaluation,kmm=False)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=r3_unlabeled_X)
        # performance4 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance4)
    l_performance1= np.array(l_performance1)
    l_performance2 = np.array(l_performance2)
    l_performance3 = np.array(l_performance3)
    l_performance4 = np.array(l_performance4)
    print(l_performance1)
    print(l_performance1.mean())
    print(l_performance1.std())
    print(l_performance2)
    print(l_performance2.mean())
    print(l_performance2.std())
    print(l_performance3)
    print(l_performance3.mean())
    print(l_performance3.std())
    print(l_performance4)
    print(l_performance4.mean())
    print(l_performance4.std())

    print(data_name, file=file)
    print(l_performance1,file=file)
    print(l_performance1.mean(),file=file)
    print(l_performance1.std(), file=file)
    print(l_performance2,file=file)
    print(l_performance2.mean(),file=file)
    print(l_performance2.std(), file=file)
    print(l_performance3,file=file)
    print(l_performance3.mean(),file=file)
    print(l_performance3.std(), file=file)
    print(l_performance4,file=file)
    print(l_performance4.mean(),file=file)
    print(l_performance4.std(), file=file)
        # print(performance5,file=file)
        # print(performance6,file=file)



