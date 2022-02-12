import copy
import random

from scipy import stats
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
import numpy as np

def normfun(x, mu, sigma):

    k=len(x)

    dis=np.expand_dims(x-mu,axis=0)

    pdf = np.exp(-0.5*dis.dot(np.linalg.inv(sigma)).dot(dis.T))/np.sqrt(((2*np.pi)**k)*np.linalg.det(sigma))

    return pdf

class SemiGMM(InductiveEstimator,ClassifierMixin):
    def __init__(self,n_class, tolerance=1e-8, max_iterations=300):
        self.n_class=n_class
        self.tolerance=tolerance
        self.max_iterations=max_iterations
        self._estimator_type = ClassifierMixin._estimator_type

    def fit(self,X,y,unlabled_X):
        # print(X)
        L=len(X)
        U=len(unlabled_X)
        m=L+U
        lable_set={}

        for _ in range(self.n_class):
            lable_set[_]=set()
        for _ in range(L):
            lable_set[y[_]].add(_)
        self.mu=[]
        self.alpha=[]

        # for i in range(self.n_class):
        #     self.alpha.append(1.0*len(lable_set[i])/L)
        #     _mu=0
        #     for item in lable_set[i]:
        #         _mu=_mu+X[item]
        #     _mu=_mu/len(lable_set[i])
        #     self.mu.append(_mu)
            # _sigma=0
            # for item in lable_set[i]:
            #     dis=X[item]-self.mu[i]
            #     if len(dis.shape)==1:
            #         dis = np.expand_dims(dis, axis=0)
            #     _sigma=_sigma+np.matmul(dis.T,dis)
            # _sigma=_sigma/len(lable_set[i])
            # self.sigma.append(_sigma)

        # self.mu=np.array(self.mu)
        # self.alpha=np.array(self.alpha)
        self.gamma=np.empty((U,self.n_class))
        self.alpha = np.random.rand(self.n_class)
        self.alpha = self.alpha / self.alpha.sum()      # 保证所有p_k的和为1
        self.mu = np.random.rand(self.n_class, X.shape[1])
        self.sigma = np.empty((self.n_class, X.shape[1], X.shape[1]))
        for i in range(self.n_class):
            self.sigma[i] = np.eye(X.shape[1])

        for _ in range(self.max_iterations):
            # E Step
            print(_)
            pre=copy.copy(self.alpha)

            for j in range(U):
                _sum=0
                for i in range(self.n_class):
                    _sum+=self.alpha[i]*normfun(unlabled_X[j],self.mu[i],self.sigma[i])
                for i in range(self.n_class):
                    self.gamma[j][i]=self.alpha[i]*normfun(unlabled_X[j],self.mu[i],self.sigma[i])/_sum
                    # print(self.gamma[j][i])


            # M step
            for i in range(self.n_class):
                _sum_mu=0
                _sum_sigma=np.zeros((X.shape[1],X.shape[1]))
                _norm=0
                _norm+=len(lable_set[i])

                for j in lable_set[i]:
                    _sum_mu+=X[j]
                for j in range(U):
                    _sum_mu+=self.gamma[j][i]*unlabled_X[j]
                    _norm+=self.gamma[j][i]
                #print(_norm)
                self.mu[i]=_sum_mu/_norm

                print(self.mu[i])
                self.alpha[i]=_norm/m


                for j in lable_set[i]:
                    _sum_sigma+=np.outer(X[j]-self.mu[i],X[j]-self.mu[i])

                for j in range(U):
                    _sum_sigma += self.gamma[j][i]*np.outer(unlabled_X[j] - self.mu[i], unlabled_X[j] - self.mu[i])
                    # print(unlabled_X[j])
                    # print(self.mu[i])
                #print(_sum_sigma)
                self.sigma[i]=_sum_sigma/_norm

            isOptimal = True
            for i in range(self.n_class):
                if abs((self.alpha[i] - pre[i])/pre[i])>self.tolerance:
                    isOptimal=False

            if isOptimal:
                break

        return self

    def predict_proba(self,X):
        proba=np.empty((len(X),self.n_class))
        for i in range(len(X)):
            _sum=0
            for j in range(self.n_class):
                _sum+=normfun(X[i],self.mu[j],self.sigma[j])
            for j in range(self.n_class):
                proba[i][j]=normfun(X[i],self.mu[j],self.sigma[j])/_sum
        return proba

    def predict(self,X):
        proba=self.predict_proba(X)
        result=np.zeros(len(X))
        for _ in range(len(X)):
            result[_]=np.argmax(proba[_])
        return result





