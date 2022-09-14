from LAMDA_SSL.Algorithm.Regression.CoReg import CoReg
from LAMDA_SSL.Algorithm.Regression.ICTReg import ICTReg
from LAMDA_SSL.Algorithm.Regression.PiModelReg import PiModelReg
from LAMDA_SSL.Algorithm.Regression.MeanTeacherReg import MeanTeacherReg
from LAMDA_SSL.utils import class_status
from sklearn.preprocessing import QuantileTransformer,OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import ColumnTransformer
from LAMDA_SSL.Split.DataSplit import DataSplit
import pickle

path='../../data/numerical_only/regression/'

log_transform=[False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                False,
                None,
                None,
                None
               ]
data_id =[197,
    201,
    216,
    300,
    287,
    296,
    537,
    574,
    42225,
    42688,
    42712,
    42729,
    42731,
    23515,
    42720,
    43093,
    43174,
     None,
     None,
     None,
     ]
dataset=["cpu_act",
         "pol",
         "elevators",
         "isolet",
         "wine_quality",
          "Ailerons",
          "houses",
          "house_16H",
          "diamonds",
          "Brazilian_houses",
          "Bike_Sharing_Demand",
          "nyc-taxi-green-dec-2016",
          "house_sales",
          "sulfur",
          "medical_charges",
          "MiamiHousing2016",
          "superconduct",
          "california",
          "fifa",
         #"year"
         ]

from LAMDA_SSL.Evaluation.Regressor.Mean_Absolute_Error import Mean_Absolute_Error
from LAMDA_SSL.Evaluation.Regressor.Mean_Squared_Error import Mean_Squared_Error
from LAMDA_SSL.Evaluation.Regressor.Mean_Squared_Log_Error import Mean_Squared_Log_Error

evaluation={
    'Mean_Absolute_Error':Mean_Absolute_Error(),
    'Mean_Squared_Error':Mean_Squared_Error(),
    'Mean_Squared_Log_Error':Mean_Squared_Log_Error()
}

for _ in range(len(dataset)):
    data_name=dataset[_]
    log=log_transform[_]
    id=data_id[_]
    data_path= path+'data_'+data_name
    with open(data_path,"rb") as f:
        data = pickle.load(f)
    X,y=data
    test_X,test_y,train_X,train_y=DataSplit(stratified=False,shuffle=True,random_state=0, X=X, y=y,size_split=0.2)
    labeled_X,labeled_y,unlabeled_X,unlabeled_y=DataSplit(stratified=False,shuffle=True,random_state=0, X=train_X, y=train_y,size_split=0.375)
    coreg_model=CoReg(evaluation=evaluation)
    coreg_model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)
    performance = coreg_model.evaluate(X=test_X, y=test_y)
    print(performance)
    # print(class_status(y).num_classes)
    # print(class_status(y).class_counts)



