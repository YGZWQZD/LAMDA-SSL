import numpy as np

from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Algorithm.Classification.SSGMM import SSGMM
from LAMDA_SSL.Algorithm.Classification.TSVM import TSVM
from LAMDA_SSL.Algorithm.Classification.LapSVM import LapSVM
from LAMDA_SSL.Algorithm.Classification.LabelPropagation import LabelPropagation
from LAMDA_SSL.Algorithm.Classification.LabelSpreading import LabelSpreading
from LAMDA_SSL.Algorithm.Classification.Co_Training import Co_Training
from LAMDA_SSL.Algorithm.Classification.Tri_Training import Tri_Training
from LAMDA_SSL.Algorithm.Classification.Assemble import Assemble
from LAMDA_SSL.Algorithm.Classification.SemiBoost import SemiBoost
import csv
import pickle
from LAMDA_SSL.Split.DataSplit import DataSplit
import copy
from sklearn.preprocessing import QuantileTransformer,StandardScaler

datasets=[
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

algorithms={'SSGMM':SSGMM(),'TSVM':TSVM(),'LapSVM':LapSVM(),'LabelPropagation':LabelPropagation(),
           'LabelSpreading':LabelSpreading(),'Co_Training':Co_Training(),'Tri_Training':Tri_Training(),
           'Assemble':Assemble(),'SemiBoost':SemiBoost()}
Transductive=['TSVM','LabelPropagation','LabelSpreading']

path='../../data/numerical_only/balanced/'
for dataset in datasets:
        f=open(dataset+".csv", "w", encoding="utf-8")
        r = csv.DictWriter(f,['algorithm','mean','std'])
        data_path = path + 'data_' + dataset
        with open(data_path, "rb") as f:
                data = pickle.load(f)
        X, y = data


        for name,algorithm in algorithms.items():
                print(name)
                performance_list=[]
                for _ in range(5):
                        print(_)
                        test_X, test_y, train_X, train_y = DataSplit(stratified=True, shuffle=True,
                                                                     random_state=_, X=X, y=y, size_split=0.2)
                        trans = StandardScaler().fit(train_X)
                        test_X = trans.transform(test_X)
                        train_X=trans.transform(train_X)
                        _train_X = copy.deepcopy(train_X)
                        _train_y = copy.deepcopy(train_y)
                        labeled_X, labeled_y, unlabeled_X, unlabeled_y = DataSplit(stratified=True, shuffle=True,
                                                                                   random_state=_, X=_train_X,
                                                                                   y=_train_y, size_split=0.375)
                        if name in Transductive:
                                pred_y = algorithm.fit(labeled_X, labeled_y, unlabeled_X).predict(test_X,Transductive=False)
                        else:
                                pred_y=algorithm.fit(labeled_X,labeled_y,unlabeled_X).predict(test_X)
                        performance=Accuracy().scoring(test_y,pred_y)
                        performance_list.append(performance)
                        print(performance)
                performance_list=np.array(performance_list)
                mean=performance_list.mean()
                std=performance_list.std()
                d={}
                d['algorithm']=name
                d['mean']=mean
                d['std']=std
                print(d)
                r.writerow(d)
                f.close()




