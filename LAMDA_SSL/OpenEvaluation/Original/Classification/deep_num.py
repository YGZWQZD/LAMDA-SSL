from LAMDA_SSL.Algorithm.Classification.LadderNetwork import LadderNetwork
from LAMDA_SSL.Algorithm.Classification.PiModel import PiModel
from LAMDA_SSL.Algorithm.Classification.TemporalEnsembling import TemporalEnsembling
from LAMDA_SSL.Algorithm.Classification.UDA import UDA
from LAMDA_SSL.Algorithm.Classification.PseudoLabel import PseudoLabel
from LAMDA_SSL.Algorithm.Classification.S4L import S4L
from LAMDA_SSL.Algorithm.Classification.ImprovedGAN import ImprovedGAN
from LAMDA_SSL.Algorithm.Classification.SSVAE import SSVAE
from LAMDA_SSL.Algorithm.Classification.ICT import ICT
from LAMDA_SSL.Algorithm.Classification.MixMatch import MixMatch
from LAMDA_SSL.Algorithm.Classification.ReMixMatch import ReMixMatch
from LAMDA_SSL.Algorithm.Classification.FixMatch import FixMatch
from LAMDA_SSL.Algorithm.Classification.FlexMatch import FlexMatch
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Transform.ToTensor import ToTensor
from LAMDA_SSL.Dataloader.LabeledDataloader import LabeledDataLoader
from LAMDA_SSL.Dataloader.UnlabeledDataloader import UnlabeledDataLoader
from LAMDA_SSL.Sampler.RandomSampler import RandomSampler
from LAMDA_SSL.Sampler.SequentialSampler import SequentialSampler
from LAMDA_SSL.Augmentation.Tabular.Noise import Noise
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy

import csv
import pickle
from LAMDA_SSL.Split.DataSplit import DataSplit
import copy
from sklearn.preprocessing import QuantileTransformer,StandardScaler
import numpy as np

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

evaluation= Accuracy()

labeled_dataset=LabeledDataset(transform=ToTensor())
unlabeled_dataset=UnlabeledDataset(transform=ToTensor())
test_dataset=UnlabeledDataset(transform=ToTensor())

labeled_dataloader=LabeledDataLoader(batch_size=64,num_workers=0,drop_last=True)
unlabeled_dataloader=UnlabeledDataLoader(num_workers=0,drop_last=True)
test_dataloader=UnlabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)

labeled_sampler=RandomSampler(replacement=True,num_samples=64*(4000))
unlabeled_sampler=RandomSampler(replacement=True)
test_sampler=SequentialSampler()

weak_augmentation=Noise()
strong_augmentation=Noise(0.3)
augmentation={
    'weak_augmentation':weak_augmentation,
    'strong_augmentation':strong_augmentation
}


algorithms={'LadderNetwork':LadderNetwork(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset,test_dataset=test_dataset,device='cuda:0'),
           'PiModel':PiModel(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset,test_dataset=test_dataset,device='cuda:0',augmentation=weak_augmentation),
           'TemporalEnsembling':TemporalEnsembling(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset,test_dataset=test_dataset,device='cuda:0',augmentation=weak_augmentation),
           'UDA':UDA(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset,test_dataset=test_dataset,device='cuda:0',augmentation=augmentation),
           'PseudoLabel':PseudoLabel(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset,test_dataset=test_dataset,device='cuda:0',augmentation=weak_augmentation),
           # 'S4L':S4L(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset,test_dataset=test_dataset,device='cuda:0'),
           'ImprovedGAN':ImprovedGAN(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset,test_dataset=test_dataset,device='cuda:0'),
           'SSVAE':SSVAE(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset,test_dataset=test_dataset,device='cuda:0'),
           'ICT':ICT(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset,test_dataset=test_dataset,device='cuda:0',augmentation=weak_augmentation),
           'MixMatch':MixMatch(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset,test_dataset=test_dataset,device='cuda:0',augmentation=weak_augmentation),
           'ReMixMatch':ReMixMatch(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset,test_dataset=test_dataset,device='cuda:0',augmentation=augmentation),
           'FixMatch':FixMatch(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset,test_dataset=test_dataset,device='cuda:0',augmentation=augmentation),
           'FlexMatch':FlexMatch(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset,test_dataset=test_dataset,device='cuda:0',augmentation=augmentation)}

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