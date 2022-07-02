from lamda_ssl.Algorithm.Regressor.CoReg import CoReg
from lamda_ssl.Evaluation.Regression.Mean_Absolute_Error import Mean_Absolute_Error
from lamda_ssl.Evaluation.Regression.Mean_Squared_Error import Mean_Squared_Error
from lamda_ssl.Evaluation.Regression.Mean_Squared_Log_Error import Mean_Squared_Log_Error
from lamda_ssl.Dataset.Table.Boston import Boston
import numpy as np

file = open("../Result/CoReg_Boston.txt", "w")

dataset=Boston(labeled_size=0.3,test_size=0.1,stratified=False,shuffle=True,random_state=0,default_transforms=True)

labeled_X=dataset.labeled_X
labeled_y=dataset.labeled_y
unlabeled_X=dataset.unlabeled_X
unlabeled_y=dataset.unlabeled_y
test_X=dataset.test_X
test_y=dataset.test_y

# Pre_transform
pre_transform=dataset.pre_transform
pre_transform.fit(np.vstack([labeled_X, unlabeled_X]))

labeled_X=pre_transform.transform(labeled_X)
unlabeled_X=pre_transform.transform(unlabeled_X)
test_X=pre_transform.transform(test_X)

evaluation={
    'Mean_Absolute_Error':Mean_Absolute_Error(),
    'Mean_Squared_Error':Mean_Squared_Error(),
    'Mean_Squared_Log_Error':Mean_Squared_Log_Error()
}

model=CoReg(evaluation=evaluation,verbose=True,file=file)

model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)

performance=model.evaluate(X=test_X,y=test_y)

result=model.y_pred

print(result,file=file)

print(performance,file=file)
