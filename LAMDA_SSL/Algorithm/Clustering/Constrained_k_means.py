import numpy as np
from sklearn.base import ClusterMixin
from LAMDA_SSL.Base.TransductiveEstimator import TransductiveEstimator
import random
from torch.utils.data.dataset import Dataset
import LAMDA_SSL.Config.Constrained_k_means as config

class Constrained_k_means(TransductiveEstimator,ClusterMixin):
    def __init__(self,k=config.k, tolerance=config.tolerance, max_iterations=config.max_iterations,
                 evaluation=config.evaluation,verbose=config.verbose,file=config.file):
        # >> Parameter
        # >> - k: The k value for the k-means clustering algorithm.
        # >> - tolerance: Tolerance of iterative convergence.
        # >> - max_iterations: The maximum number of iterations.
        self.k=k
        self.tolerance=tolerance
        self.max_iterations=max_iterations
        self.y=None
        self.evaluation = evaluation
        self.verbose = verbose
        self.file = file
        self.X=None
        self.y_pred=None
        self._estimator_type = ClusterMixin._estimator_type

    def fit(self,X,y=None,unlabeled_X=None,cl=None,ml=None):
        if unlabeled_X is not None:
            ml = []
            cl = []
            for i in range(X.shape[0]):
                for j in range(i + 1, X.shape[0]):
                    if y[i] == y[j]:
                        ml.append({i, j})
                    else:
                        cl.append({i, j})
            X = np.vstack((X, unlabeled_X))
        self.X=X

        self.ml={}
        self.cl={}
        for _ in range(len(X)):
            self.ml[_]=set()
            self.cl[_]=set()
        constrain=np.ones((len(X),len(X)))*-1

        for _ in range(len(cl)):
            for item1 in cl[_]:
                for item2 in cl[_]:
                    if item1!=item2:
                        constrain[item1][item2]=0

        for _ in range(len(ml)):
            for item1 in ml[_]:
                for item2 in ml[_]:
                    if item1 != item2:
                        constrain[item1][item2] = -1

        for k in range(len(X)):
            for i in range(len(X)):
                for j in range(len(X)):
                    if i==k or j==k or i==j:
                        break
                    if constrain[i][k]==1 and constrain[j][k]==1:
                        if constrain[i][j]==-1:
                            constrain[i][j]=constrain[j][i]=1
                        elif constrain[i][j]==0:
                            raise ValueError('Can not satisfy constraints.')
                    if constrain[i][k]==1 and constrain[j][k]==-1:
                        if constrain[i][j]==-1:
                            constrain[i][j]=constrain[j][i]=0
                        elif constrain[i][j] == 1:
                            raise ValueError('Can not satisfy constraints.')
                    if constrain[i][k] == -1 and constrain[j][k] == 1:
                        if constrain[i][j] == -1:
                            constrain[i][j] = constrain[j][i] = 0
                        elif constrain[i][j] == 1:
                            raise ValueError('Can not satisfy constraints.')

        for i in range(len(X)):
            for j in range(len(X)):
                if constrain[i][j]==0:
                    self.cl[i].add(j)
                if constrain[i][j]==1:
                    self.ml[i].add(j)

        while True:
            Find_answer=True
            index_list = list(range(len(X)))

            random.shuffle(index_list)

            c = X[index_list[:self.k]]


            for _ in range(self.max_iterations):

                self.is_clustered = np.array([-1]*len(X))

                self.clusters = {}
                for i in range(self.k):
                    self.clusters[i] = set()

                for x_index in range(len(X)):

                    distances = {center_index: np.linalg.norm(X[x_index] - c[center_index])for center_index in range(len(c))}

                    sorted_distances = sorted(distances.items(), key=lambda kv: kv[1])

                    empty_flag = True

                    for center_index,dis_value in sorted_distances:

                        vc_result = self.violate_constraints(x_index, center_index, self.ml, self.cl)

                        if not vc_result:
                            self.clusters[center_index].add(x_index)
                            self.is_clustered[x_index] = center_index
                            for j in self.ml[x_index]:
                                self.is_clustered[j] = center_index
                            empty_flag = False
                            break

                    if empty_flag:
                        Find_answer=False
                        break
                        # raise Exception("Clustering Not Found Exception")
                if Find_answer is False:
                    break
                previous = c

                for _center in self.clusters:
                    lst = []
                    for index_value in self.clusters[_center]:
                        lst.append(X[index_value])
                    avgArr = np.array(lst)

                    if len(lst) != 0:
                        c[_center] = np.average(avgArr, axis = 0)

                isOptimal = True
                for centroid in range(len(c)):
                    original_centroid = previous[centroid]
                    curr = c[centroid]
                    if np.sum((curr - original_centroid)/original_centroid) > self.tolerance:
                        isOptimal = False
                if isOptimal:
                    break
            if Find_answer:
                break
        self.center=c
        self.y=self.is_clustered

        return self


    def violate_constraints(self, data_index, cluster_index, ml, cl):
        # print(self.clusters)
        for i in ml[data_index]:
            if self.is_clustered[i] != -1 and self.is_clustered[i] != cluster_index:
                return True

        for i in cl[data_index]:
            if self.is_clustered[i] != -1 and self.is_clustered[i] == cluster_index:
                # print('cl')
                # print(i)
                return True
        return False

    def predict(self, X=None,Transductive=True):
        if Transductive:
            result=self.y
        else:
            result=np.array([])
            for _ in range(len(X)):
                distances = np.array([np.linalg.norm(X[_] - self.center[centroid]) for centroid in range(len(self.center))])
                result = np.hstack([result,np.array([np.argmin(distances)])])
        return  result

    def evaluate(self,X=None,y=None,Transductive=True):

        if isinstance(X,Dataset) and y is None:
            y=getattr(X,'y')

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