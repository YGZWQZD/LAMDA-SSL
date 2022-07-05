import copy
import numbers
import numpy as np
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
from torch.utils.data.dataset import Dataset
from sklearn.neighbors import KNeighborsClassifier
import LAMDA_SSL.Config.Assemble as config

class Assemble(InductiveEstimator,ClassifierMixin):
    def __init__(self, base_estimater=config.base_estimater,T=config.T,alpha=config.alpha,
                 beta=config.beta,evaluation=config.evaluation,verbose=config.verbose,file=config.file):
        self.base_estimater=base_estimater
        self.T=T
        self.alpha=alpha
        self.beta=beta
        self.KNN=KNeighborsClassifier(n_neighbors=1)
        self.evaluation = evaluation
        self.verbose=verbose
        self.file=file
        self.w=[]
        self.f=[]
        self.y_pred=None
        self.y_score=None
        self._estimator_type = ClassifierMixin._estimator_type

    def predict_proba(self,X):
        y_proba=0
        for _ in range(len(self.w)):
            y_proba=y_proba+self.w[_]*self.f[_].predict_proba(X)

        return y_proba

    def predict(self,X):
        y_proba=self.predict_proba(X)
        y_pred=np.argmax(y_proba, axis=1)
        return y_pred

    def fit(self,X,y,unlabeled_X):
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
        classfier=copy.deepcopy(self.base_estimater)
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
            sample_weight=alpha*np.exp(-logits)
            sample_weight=sample_weight/sample_weight.sum()
            idx_sample=np.random.choice(l+u,l,False,p=sample_weight.tolist())
            X_sample=X_all[idx_sample]
            y_sample=y_all[idx_sample]
            sample_weight_sample=sample_weight[idx_sample]
            classfier=copy.deepcopy(self.base_estimater)
            classfier.fit(X_sample,y_sample,sample_weight_sample)

        return self

    def evaluate(self,X,y=None):

        if isinstance(X,Dataset) and y is None:
            y=getattr(X,'y')

        self.y_score = self.predict_proba(X)
        self.y_pred=self.predict(X)


        if self.evaluation is None:
            return None
        elif isinstance(self.evaluation,(list,tuple)):
            result=[]
            for eval in self.evaluation:
                score=eval.scoring(y,self.y_pred,self.y_score)
                if self.verbose:
                    print(score, file=self.file)
                result.append(score)
            self.result = result
            return result
        elif isinstance(self.evaluation,dict):
            result={}
            for key,val in self.evaluation.items():

                result[key]=val.scoring(y,self.y_pred,self.y_score)

                if self.verbose:
                    print(key,' ',result[key],file=self.file)
                self.result = result
            return result
        else:
            result=self.evaluation.scoring(y,self.y_pred,self.y_score)
            if self.verbose:
                print(result, file=self.file)
            self.result=result
            return result



