from LAMDA_SSL.Algorithm.Regression.CoReg import CoReg
from LAMDA_SSL.Algorithm.Regression.ICTReg import ICTReg
from LAMDA_SSL.Algorithm.Regression.PiModelReg import PiModelReg
from LAMDA_SSL.Algorithm.Regression.MeanTeacherReg import MeanTeacherReg
from LAMDA_SSL.utils import class_status
from sklearn.preprocessing import QuantileTransformer,OneHotEncoder,StandardScaler
from LAMDA_SSL.Algorithm.Classification.Co_Training import Co_Training
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import ColumnTransformer
from KMM import KMM
from sklearn.metrics.pairwise import rbf_kernel,linear_kernel
from LAMDA_SSL.Algorithm.Classification.TSVM import TSVM
from LAMDA_SSL.Algorithm.Classification.LapSVM import LapSVM
from LAMDA_SSL.Algorithm.Classification.Assemble import Assemble
from LAMDA_SSL.Algorithm.Classification.SemiBoost import SemiBoost
from LAMDA_SSL.Algorithm.Classification.SSGMM import SSGMM
from Self_Training import Self_Training
from Pseudo_Label import Pseudo_Label
from LAMDA_SSL.Split.DataSplit import DataSplit
from LAMDA_SSL.Algorithm.Classification.Tri_Training import Tri_Training
import copy
from sklearn.linear_model._logistic import LogisticRegression
from LAMDA_SSL.Algorithm.Classification.LabelSpreading import LabelSpreading
from LAMDA_SSL.Algorithm.Classification.LabelPropagation import LabelPropagation
import pickle
from sklearn.svm import SVC
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
file = open("tritraining.txt", "w")



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

def distribution_selection_2(X,y,p1=0.80,p2=0.20):
    mean_X=np.mean(X,axis=0)
    r=np.random.random(X.shape[0])
    s = []
    for i in range(X.shape[0]):
        flag=False
        j=random.choice(list(range(X.shape[1])))
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
    r_y=y[r_s]
    r_X=X[r_s]
    # r_X= np.random.rand(s_X.shape[0],s_X.shape[1])
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
        "wine"
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
    data_path= path+'data_'+data_name
    with open(data_path,"rb") as f:
        data = pickle.load(f)
    X,y=data

    l_performance1=[]
    l_performance2=[]
    l_performance3=[]
    l_performance4=[]

    for _ in range(5):
        test_X, test_y, train_X, train_y = DataSplit(stratified=True, shuffle=True, random_state=_, X=X, y=y,
                                                     size_split=0.2)
        # test_X, test_y,rtx,rty=distribution_selection_2(test_X, test_y, p1=0.1, p2=0.9)
        trans = StandardScaler().fit(train_X)
        test_X = trans.transform(test_X)
        _train_X=copy.deepcopy(train_X)
        _train_y=copy.deepcopy(train_y)
        labeled_X,labeled_y,unlabeled_X,unlabeled_y=DataSplit(stratified=True,shuffle=True,random_state=_, X=_train_X, y=_train_y,size_split=0.01)
        # labeled_X, labeled_y, rlx, rly = distribution_selection_2(labeled_X, labeled_y,p1=0.1,p2=0.9)
        s_unlabeled_X,s_unlabeled_y,r_unlabeled_X,r_unlabeled_y=distribution_selection_2(unlabeled_X,unlabeled_y,p1=0.1,p2=0.9)
        labeled_X=trans.transform(labeled_X)
        s_unlabeled_X = trans.transform(s_unlabeled_X)
        r_unlabeled_X = trans.transform(r_unlabeled_X)

        baseline=SVC(C=1.0,kernel='rbf',probability=True,gamma='auto').fit(labeled_X,labeled_y)
        print(Accuracy().scoring(y_true=test_y,y_pred=baseline.predict(test_X)))

        ssl_model = SSGMM().fit(labeled_X,labeled_y,r_unlabeled_X)
        y_pred = ssl_model.predict(test_X)
        print(y_pred)
        print(Accuracy().scoring(y_true=test_y,y_pred=y_pred))
        ssl_model = SSGMM().fit(labeled_X,labeled_y,r_unlabeled_X)
        y_pred = ssl_model.predict(test_X)
        print(y_pred)
        print(Accuracy().scoring(y_true=test_y,y_pred=y_pred))
        # ssl_model = SemiBoost(similarity_kernel='rbf',gamma=0.01).fit(labeled_X,labeled_y,np.random.randn(r_unlabeled_X.shape[0],r_unlabeled_X.shape[1]))
        # y_pred = ssl_model.predict(test_X)
        # print(y_pred)
        # print(Accuracy().scoring(y_true=test_y,y_pred=y_pred))



        # print('kmm_pseudo')
        #
        # self_training_model = TSVM(evaluation=evaluation)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=s2_unlabeled_X)
        # performance1 = self_training_model.evaluate(X=test_X, y=test_y,Transductive=False)
        # print(performance1)
        # l_performance1.append(performance1)

        # self_training_model = Tri_Training(evaluation=evaluation)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=s2_unlabeled_X)
        # performance2 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance2)
        # l_performance2.append(performance2)

        # self_training_model = TSVM(evaluation=evaluation)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=r2_unlabeled_X)
        # performance3 = self_training_model.evaluate(X=test_X, y=test_y,Transductive=False)
        # print(performance3)
        # l_performance3.append(performance3)

        # self_training_model = Tri_Training(evaluation=evaluation)
        # self_training_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=r2_unlabeled_X)
        # performance4 = self_training_model.evaluate(X=test_X, y=test_y)
        # print(performance4)
        # l_performance4.append(performance4)

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
    # l_performance1= np.array(l_performance1)
    # l_performance2 = np.array(l_performance2)
    # l_performance3 = np.array(l_performance3)
    # l_performance4 = np.array(l_performance4)
    # print(l_performance1)
    # print(l_performance1.mean())
    # print(l_performance1.std())
    # print(l_performance2)
    # print(l_performance2.mean())
    # print(l_performance2.std())
    # print(l_performance3)
    # print(l_performance3.mean())
    # print(l_performance3.std())
    # print(l_performance4)
    # print(l_performance4.mean())
    # print(l_performance4.std())
    #
    # print(data_name, file=file)
    # print(l_performance1,file=file)
    # print(l_performance1.mean(),file=file)
    # print(l_performance1.std(), file=file)
    # print(l_performance2,file=file)
    # print(l_performance2.mean(),file=file)
    # print(l_performance2.std(), file=file)
    # print(l_performance3,file=file)
    # print(l_performance3.mean(),file=file)
    # print(l_performance3.std(), file=file)
    # print(l_performance4,file=file)
    # print(l_performance4.mean(),file=file)
    # print(l_performance4.std(), file=file)
        # print(performance5,file=file)
        # print(performance6,file=file)



