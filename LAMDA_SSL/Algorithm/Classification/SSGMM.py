import copy
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
import numpy as np
from torch.utils.data.dataset import Dataset
from LAMDA_SSL.utils import class_status
import LAMDA_SSL.Config.SSGMM as config
from scipy import stats
class SSGMM(InductiveEstimator,ClassifierMixin):
    def __init__(self,tolerance=config.tolerance, max_iterations=config.max_iterations, num_classes=config.num_classes,
                 random_state=config.random_state,evaluation=config.evaluation,verbose=config.verbose,file=config.file):
        # >> Parameter
        # >> - num_classes: The number of classes.
        # >> - tolerance: Tolerance for iterative convergence.
        # >> - max_iterations: The maximum number of iterations.

        self.num_classes=num_classes
        self.tolerance=tolerance
        self.max_iterations=max_iterations
        self.random_state=random_state
        self.evaluation = evaluation
        self.verbose = verbose
        self.file = file
        self.y_pred=None
        self.y_score=None
        self._estimator_type = ClassifierMixin._estimator_type

    def normfun(self,x, mu, sigma):
        pdf=stats.multivariate_normal(mu,sigma,allow_singular=True).pdf(x)
        return pdf

    def fit(self,X,y,unlabeled_X):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(y).num_classes
        L=len(X)
        U=len(unlabeled_X)
        m=L+U
        labele_set={}

        for _ in range(self.num_classes):
            labele_set[_]=set()
        for _ in range(L):
            labele_set[y[_]].add(_)

        self.gamma=np.empty((U,self.num_classes))
        self.alpha = np.zeros(self.num_classes)
        self.mu=np.zeros((self.num_classes,X.shape[1]))
        self.sigma=np.zeros((self.num_classes,X.shape[1],X.shape[1]))
        for i in range(self.num_classes):
            self.alpha[i]=len(labele_set[i])/L
            _sum_mu=0
            for j in labele_set[i]:
                _sum_mu+=X[j]
            self.mu[i]=_sum_mu/len(labele_set[i])
            _sum_sigma=0
            for j in labele_set[i]:
                _sum_sigma += np.outer(X[j] - self.mu[i], X[j] - self.mu[i])
            self.sigma[i]=_sum_sigma/len(labele_set[i])
        for _ in range(self.max_iterations):
            # E Step
            pre=copy.copy(self.alpha)

            for j in range(U):
                _sum=0
                for i in range(self.num_classes):
                    _sum+=self.alpha[i]*self.normfun(unlabeled_X[j],self.mu[i],self.sigma[i])
                for i in range(self.num_classes):
                    if _sum==0:
                        self.gamma[j][i]=self.alpha[i]/self.num_classes
                    else:
                        self.gamma[j][i]=self.alpha[i]*self.normfun(unlabeled_X[j],self.mu[i],self.sigma[i])/_sum
            # M step
            for i in range(self.num_classes):
                _sum_mu=0
                _sum_sigma=np.zeros((X.shape[1],X.shape[1]))
                _norm=0
                _norm+=len(labele_set[i])

                for j in labele_set[i]:
                    _sum_mu+=X[j]
                for j in range(U):
                    _sum_mu+=self.gamma[j][i]*unlabeled_X[j]
                    _norm+=self.gamma[j][i]

                self.mu[i]=_sum_mu/_norm

                self.alpha[i]=_norm/m


                for j in labele_set[i]:
                    _sum_sigma+=np.outer(X[j]-self.mu[i],X[j]-self.mu[i])

                for j in range(U):
                    _sum_sigma += self.gamma[j][i]*np.outer(unlabeled_X[j] - self.mu[i], unlabeled_X[j] - self.mu[i])
                self.sigma[i]=_sum_sigma/_norm

            isOptimal = True
            for i in range(self.num_classes):
                if abs((self.alpha[i] - pre[i])/pre[i])>self.tolerance:
                    isOptimal=False

            if isOptimal:
                break

        return self

    def predict_proba(self,X):
        y_proba=np.empty((len(X),self.num_classes))
        for i in range(len(X)):
            _sum=0
            for j in range(self.num_classes):
                _sum+=self.normfun(X[i],self.mu[j],self.sigma[j])
            for j in range(self.num_classes):
                if _sum ==0:
                    y_proba[i][j]=1/self.num_classes
                else:
                    y_proba[i][j]=self.normfun(X[i],self.mu[j],self.sigma[j])/_sum
        return y_proba

    def predict(self,X):
        y_proba=self.predict_proba(X)
        y_pred=np.argmax(y_proba, axis=1)
        return y_pred

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