import copy
import numbers

import numpy as np

from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
class Assemble(InductiveEstimator,ClassifierMixin):
    def __init__(self, base_model=SVC(probability=True),T=100,alpha=1,beta=0.9):
        self._estimator_type = ClassifierMixin._estimator_type
        self.base_model=base_model
        self.T=T
        self.alpha=alpha
        self.beta=beta
        self._estimator_type = ClassifierMixin._estimator_type
        self.KNN=KNeighborsClassifier(n_neighbors=1)
        self.w=[]
        self.f=[]

    def predict_proba(self,X):

        result=0
        for _ in range(len(self.w)):
            result=result+self.w[_]*self.f[_].predict_proba(X)

        return result

    def predict(self,X):
        probas=self.predict_proba(X)
        result=np.argmax(probas, axis=1)
        # _len=len(result)
        # result=copy.copy(result)
        # for _ in range(_len):
        #     result[_]=self.rev_class_dict[result[_]]
        return result

    def fit(self,X,y,unlabeled_X):
        # 二分类

        # classes, y_indices = np.unique(y, return_inverse=True)
        # if len(classes)!=2:
        #     raise ValueError('TSVM can only be used in binary classification.')
        # # print(classes)
        #
        # self.class_dict={classes[0]:-1,classes[1]:1}
        # self.rev_class_dict = {-1:classes[0] ,  1:classes[1]}
        # y=copy.copy(y)
        # for _ in range(X.shape[0]):
        #     y[_]=self.class_dict[y[_]]
        # y\in (-1,1)

        self.w=[]
        self.f=[]
        l=X.shape[0]
        u=unlabeled_X.shape[0]
        sample_weight=np.zeros(l+u)
        for i in range(l):
            sample_weight[i]=self.beta/l
        for i in range(u):
            sample_weight[i+l]=(1-self.beta)/u
        unlabeled_y=self.KNN.fit(X,y).predict(unlabeled_X)
        classfier=copy.deepcopy(self.base_model)
        X_all=np.concatenate((X,unlabeled_X))
        y_all=np.concatenate((y,unlabeled_y))
        classfier.fit(X_all,y_all,sample_weight=sample_weight)

        for i in range(self.T):
            self.f.append(classfier)
            _y_all=classfier.predict(X_all)
            epsilon=0
            for _ in range(l+u):
                if _y_all[_]!=y_all[_]:
                    epsilon+=sample_weight[_]
            if epsilon>0.5:
                break
            w=np.log((1-epsilon)/epsilon)*0.5
            self.w.append(w)

            probas=self.predict_proba(X_all)
            logits = np.max(probas, axis=1)
            unlabeled_y=self.predict(unlabeled_X)


            y_all=np.concatenate((y,unlabeled_y))
            if isinstance(self.alpha,numbers.Number):
                alpha=np.ones(l+u)*self.alpha
            else:
                alpha=self.alpha
            # print(logits.shape)
            sample_weight=alpha*np.exp(-logits)
            sample_weight=sample_weight/sample_weight.sum()
            idx_sample=np.random.choice(l+u,l,False,p=sample_weight.tolist())
            X_sample=X_all[idx_sample]
            y_sample=y_all[idx_sample]
            sample_weight_sample=sample_weight[idx_sample]
            classfier=copy.deepcopy(self.base_model)
            classfier.fit(X_sample,y_sample,sample_weight_sample)
            # self.f.append(classfier)

        return self




