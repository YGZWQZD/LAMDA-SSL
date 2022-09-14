import copy
from torch.utils.data.dataset import Dataset
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
import numpy as np
from KMM import KMM
from sklearn.svm import SVC

class Pseudo_Label(InductiveEstimator,ClassifierMixin):
    def __init__(self,
                 base_estimator=SVC(C=1.0,kernel='rbf',probability=True,gamma='auto'),
                 threshold=None,
                 evaluation=None, verbose=False, file=None,kmm=False
                 ):
        self.base_estimator=base_estimator
        self.threshold=threshold
        self.evaluation=evaluation
        self.verbose=verbose
        self.file=file
        self.kmm=kmm
        self._estimator_type = ClassifierMixin._estimator_type

    def fit(self,X=None,y=None,unlabeled_X=None):
        print('fit')
        self.f=copy.deepcopy(self.base_estimator)
        self.h=copy.deepcopy(self.base_estimator)
        num_labeled = X.shape[0]
        num_unlabeled = unlabeled_X.shape[0]
        print(np.ones(num_labeled,dtype=float).shape)


        if self.kmm:
            beta_f = KMM().fit(unlabeled_X, X).squeeze(axis=-1)
            beta_h = KMM().fit(X, unlabeled_X).squeeze(axis=-1)
            beta_f =np.concatenate((np.ones(num_labeled,dtype=float), beta_f))
            beta_h= np.concatenate((beta_h,np.ones(num_unlabeled,dtype=float)))
        else:
            beta_f=np.ones(num_labeled+num_unlabeled,dtype=float)
            beta_h=np.ones(num_labeled+num_unlabeled,dtype=float)
        classes, y_indices = np.unique(y, return_inverse=True)
        idx=np.arange(num_labeled+num_unlabeled)

        self.class_dict={classes[0]:-1,classes[1]:1}
        self.rev_class_dict = {-1:classes[0] ,  1:classes[1]}
        y=copy.copy(y)
        for _ in range(X.shape[0]):
            y[_]=self.class_dict[y[_]]

        idx_label=idx[:num_labeled]
        idx_not_label=idx[num_labeled:]
        X_all=np.concatenate((X,unlabeled_X))
        y_all=np.concatenate((y,np.zeros(num_unlabeled,dtype=int)))
        self.h.fit(X,y,sample_weight=beta_h[:num_labeled])
        prob = self.h.predict_proba(unlabeled_X)
        pred = self.h.classes_[np.argmax(prob, axis=1)]
        y_all[idx_not_label]=pred
        max_proba = np.max(prob, axis=1)
        if self.threshold is None:
            self.threshold=0
        selected = max_proba > self.threshold
        new_idx_label=idx_not_label[selected]
        idx_label=np.concatenate((idx_label,new_idx_label))
        self.f.fit(X_all[idx_label],y_all[idx_label],sample_weight=beta_f[idx_label])
        return self

    def predict(self,X=None,valid=None):
        print('predict')
        prob=self.predict_proba(X=X,valid=valid)
        pred = self.f.classes_[np.argmax(prob, axis=1)]
        for _ in range(X.shape[0]):
            pred[_]=self.rev_class_dict[pred[_]]
        return pred

    def predict_proba(self,X=None,valid=None):
        prob = self.f.predict_proba(X)
        return prob

    def evaluate(self,X,y=None):

        if isinstance(X,Dataset) and y is None:
            y=getattr(X,'y')

        self.y_score = self.predict_proba(X)
        self.y_pred=self.predict(X)


        if self.evaluation is None:
            return None
        elif isinstance(self.evaluation,(list,tuple)):
            performance=[]
            for eval in self.evaluation:
                score=eval.scoring(y,self.y_pred,self.y_score)
                if self.verbose:
                    print(score, file=self.file)
                performance.append(score)
            self.performance = performance
            return performance
        elif isinstance(self.evaluation,dict):
            performance={}
            for key,val in self.evaluation.items():

                performance[key]=val.scoring(y,self.y_pred,self.y_score)

                if self.verbose:
                    print(key,' ',performance[key],file=self.file)
                self.performance = performance
            return performance
        else:
            performance=self.evaluation.scoring(y,self.y_pred,self.y_score)
            if self.verbose:
                print(performance, file=self.file)
            self.performance=performance
            return performance


