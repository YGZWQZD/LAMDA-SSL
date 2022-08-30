import numpy as np
from sklearn import neighbors
import copy
from sklearn.metrics.pairwise import rbf_kernel,linear_kernel
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin

from torch.utils.data.dataset import Dataset
import LAMDA_SSL.Config.SemiBoost as config

class SemiBoost(InductiveEstimator,ClassifierMixin):
    # Binary
    def __init__(self, base_estimator = config.base_estimator,
                        n_neighbors=config.n_neighbors, n_jobs = config.n_jobs,
                        T = config.T,
                        sample_percent = config.sample_percent,
                        sigma_percentile = config.sigma_percentile,
                        similarity_kernel = config.similarity_kernel,gamma=config.gamma,
                        evaluation=config.evaluation,verbose=config.verbose,file=config.file):
        # >> Parameter:
        # >> - base_estimator: The base supervised learner used in the algorithm.
        # >> - similarity_kernel: 'rbf'、'knn' or callable. Specifies the kernel type to be used in the algorithm.
        # >> - n_neighbors: It is valid when the kernel function is 'knn', indicating the value of k in the k nearest neighbors.
        # >> - n_jobs: It is valid when the kernel function is 'knn', indicating the number of parallel jobs.
        # >> - gamma: It is valid when the kernel function is 'rbf', indicating the gamma value of the rbf kernel.
        # >> - T: the number of base learners. It is also the number of iterations.
        # >> - sample_percent: The number of samples sampled at each iteration as a proportion of the remaining unlabeled samples.
        # >> - sigma_percentile: Scale parameter used in the 'rbf' kernel.
        self.base_estimator = base_estimator
        self.n_neighbors=n_neighbors
        self.n_jobs=n_jobs
        self.T=T
        self.sample_percent=sample_percent
        self.sigma_percentile=sigma_percentile
        self.similarity_kernel=similarity_kernel
        self.gamma=gamma
        self.evaluation = evaluation
        self.verbose = verbose
        self.file = file
        self.y_pred=None
        self.y_score=None
        self._estimator_type = ClassifierMixin._estimator_type

    def fit(self, X, y,unlabeled_X):
        classes, y_indices = np.unique(y, return_inverse=True)

        self.class_dict={classes[0]:-1,classes[1]:1}
        self.rev_class_dict = {-1:classes[0] ,  1:classes[1]}
        y=copy.copy(y)
        for _ in range(X.shape[0]):
            y[_]=self.class_dict[y[_]]

        num_labeled=X.shape[0]
        num_unlabeled=unlabeled_X.shape[0]

        idx=np.arange(num_labeled+num_unlabeled)
        idx_labeled=idx[:num_labeled]
        idx_unlabeled=idx[num_labeled:]

        X_all=np.concatenate((X,unlabeled_X))
        y_all=np.concatenate((y,np.zeros(num_unlabeled,dtype=int)))

        if self.similarity_kernel == 'knn':
            self.S = neighbors.kneighbors_graph(X_all,
                                                n_neighbors=self.n_neighbors,
                                                mode='distance',
                                                include_self=True,
                                                n_jobs=self.n_jobs)
            self.S = np.asarray(self.S.todense())
        elif self.similarity_kernel == 'linear':
            self.S = linear_kernel(X_all, X_all)
        elif self.similarity_kernel == 'rbf':
            self.S=rbf_kernel(X_all,X_all,self.gamma)
        elif callable(self.similarity_kernel):
            if self.gamma is not None:
                self.S = self.similarity_kernel(X_all,X_all,gamma=self.gamma)
            else:
                self.S = self.similarity_kernel(X_all,X_all)
        else:
            self.S=rbf_kernel(X_all,X_all,self.gamma)

        self.models = []
        self.weights = []
        H = np.zeros(num_unlabeled)
        for t in range(self.T):
            p = np.dot(self.S[:,idx_labeled], (y_all[idx_labeled]==1))[idx_unlabeled]*np.exp(-2*H)+\
                np.dot(self.S[:,idx_unlabeled], np.exp(H))[idx_unlabeled]*np.exp(-H)
            q = np.dot(self.S[:,idx_labeled], (y_all[idx_labeled]==-1))[idx_unlabeled]*np.exp(2*H)+\
                np.dot(self.S[:,idx_unlabeled], np.exp(-H))[idx_unlabeled]*np.exp(H)

            z = np.sign(p-q)
            confidence = np.abs(p-q)
            sample_weights = confidence / np.sum(confidence)
            if np.any(sample_weights != 0):
                idx_select = np.random.choice(np.arange(len(z)),
                                              size = int(self.sample_percent*len(idx_unlabeled)),
                                              p = sample_weights,
                                              replace = False)
                idx_sample = idx_unlabeled[idx_select]
            else:
                break

            idx_labeled = np.concatenate([idx_labeled,idx_sample])
            X_t = X_all[idx_labeled,:]
            y_all[idx_sample]= z[idx_select]
            y_t = y_all[idx_labeled]

            base_estimator = self.base_estimator
            base_estimator.fit(X_t, y_t)
            h = base_estimator.predict(X_all[idx_unlabeled])
            idx_unlabeled = np.array([i for i in np.arange(len(y_all)) if i not in idx_labeled])

            e = (np.dot(p,h==-1) + np.dot(q,h==1))/(np.sum(np.add(p,q)))
            a = 0.25*np.log((1-e)/e)
            if a<0:
                break
            self.models.append(base_estimator)
            self.weights.append(a)
            H = np.zeros(len(idx_unlabeled))
            for i in range(len(self.models)):
                H = np.add(H, self.weights[i]*self.models[i].predict(X_all[idx_unlabeled]))

            if len(idx_unlabeled) == 0:
                break
        self.unlabeled_X=unlabeled_X
        self.unlabeled_y=y_all[num_unlabeled:]
        return self

    def predict_proba(self, X):
        y_proba = np.full((X.shape[0], 2), 0, np.float)
        for i in range(len(self.models)):
            y_proba = np.add(y_proba, self.weights[i] * self.models[i].predict_proba(X))
        return y_proba

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(len(self.models)):
            y_pred = np.add(y_pred, self.weights[i]*self.models[i].predict(X))
        y_pred = np.array(list(1 if x>0 else -1 for x in y_pred))
        y_pred = y_pred.astype(int)
        for _ in range(X.shape[0]):
            y_pred[_]=self.rev_class_dict[y_pred[_]]
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