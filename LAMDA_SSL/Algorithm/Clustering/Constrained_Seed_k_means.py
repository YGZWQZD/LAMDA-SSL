import numpy as np
from sklearn.base import ClusterMixin
from LAMDA_SSL.Base.TransductiveEstimator import TransductiveEstimator
import copy
from torch.utils.data.dataset import Dataset
import LAMDA_SSL.Config.Constrained_Seed_k_means as config

class Constrained_Seed_k_means(TransductiveEstimator, ClusterMixin):
    def __init__(self, k=config.k, tolerance=config.tolerance, max_iterations=config.max_iterations,evaluation=config.evaluation,
                 verbose=config.verbose,file=config.file):
        # >> Parameter
        # >> - k: The k value for the k-means clustering algorithm.
        # >> - tolerance: Tolerance of iterative convergence.
        # >> - max_iterations: The maximum number of iterations.
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.evaluation = evaluation
        self.verbose = verbose
        self.file = file
        self.X=None
        self.y_pred=None
        self._estimator_type = ClusterMixin._estimator_type

    def fit(self, X, y=None, unlabeled_X=None,clusters=None):
        assert y is not None or clusters is not None
        if clusters is None:
            clusters = {}
            for _ in range(self.k):
                clusters[_] = set()
            for _ in range(len(X)):
                clusters[y[_]].add(_)

        c=[]
        for _ in range(self.k):
            s=clusters[_]
            l_set=len(s)
            if l_set==0:
                raise ValueError('Set is empty!')
            sum=0
            for idx in s:
                sum+=X[idx]
            sum=sum/l_set
            c.append(sum)
        c=np.array(c)

        _X=np.vstack([X,unlabeled_X])

        self.X=_X

        for i in range(self.max_iterations):

            self.clusters = copy.copy(clusters)

            self.unlabeled=[True]*len(_X)

            self.is_clustered=np.array([-1]*len(_X))

            for _ in range(self.k):
                s = self.clusters[_]
                for idx in s:
                    self.is_clustered[idx]=_
                    self.unlabeled[idx]=False

            unlabeled_idx=np.arange(len(_X))
            self.unlabeled_set=unlabeled_idx[self.unlabeled]

            for x_index in self.unlabeled_set:

                distances = np.array([np.linalg.norm(_X[x_index] - c[centroid]) for centroid in range(len(c))])
                r=np.argmin(distances).item()

                self.clusters[r].add(x_index)
                self.is_clustered[x_index]=r

            previous = c

            for _center in self.clusters:
                lst = []
                for index_value in self.clusters[_center]:
                    lst.append(_X[index_value])
                avgArr = np.array(lst)

                if len(lst) != 0:
                    c[_center] = np.average(avgArr, axis=0)

            isOptimal = True
            for centroid in range(len(c)):
                original_centroid = previous[centroid]
                curr = c[centroid]
                if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    isOptimal = False
            if isOptimal:
                break

        self.center = c
        self.y = self.is_clustered
        return self

    def predict(self, X=None, Transductive=True):
        if Transductive:
            result = self.y
        else:
            result = np.array([])
            for _ in range(len(X)):
                distances = np.array([np.linalg.norm(X[_] - self.center[centroid]) for centroid in range(len(self.center))])
                result = np.hstack([result, np.argmin(distances)])
        return result

    def evaluate(self, X=None, y=None,Transductive=True):
        if isinstance(X, Dataset) and y is None:
            y= getattr(X, 'y')

        self.y_pred=self.predict(X,Transductive=Transductive)

        if Transductive:
            X=self.X

        if self.evaluation is None:
            return None

        elif isinstance(self.evaluation,(list,tuple)):
            performance=[]
            for eval in self.evaluation:
                score=eval.scoring(y,self.y_pred,X)
                if self.verbose:
                    print(score, file=self.file)
                performance.append(score)
            self.performance = performance
            return performance
        elif isinstance(self.evaluation,dict):
            performance={}
            for key,val in self.evaluation.items():
                performance[key]=val.scoring(y,self.y_pred,X)
                if self.verbose:
                    print(key,' ',performance[key],file=self.file)
                self.performance = performance
            return performance
        else:
            performance=self.evaluation.scoring(y,self.y_pred,X)
            if self.verbose:
                print(performance, file=self.file)
            self.performance=performance
            return performance