import numpy as np
from sklearn.base import ClusterMixin
from Semi_sklearn.Base.TransductiveEstimator import TransductiveEstimator
import copy


class Constrained_Seed_k_means(TransductiveEstimator, ClusterMixin):
    def __init__(self, k, tolerance=0.00001, max_iterations=300):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, X, y=None, unlabled_X=None,clusters=None):
        # index_list = list(range(len(X)))
        # random.shuffle(index_list)
        #
        # c = X[index_list[:self.k]]
        assert clusters is not None

        c=np.array([])
        for _ in range(self.k):
            s=clusters[_]
            l_set=len(set)
            if l_set==0:
                raise ValueError('Set is empty!')
            sum=0
            for idx in s:
                sum+=X[idx]
            sum=sum/l_set
            c=np.vstack([c,sum])




        for i in range(self.max_iterations):


            self.clusters = copy.copy(clusters)

            self.unlabled=np.ones(len(X))

            self.is_clustered=np.ones(len(X))*-1

            for _ in range(self.k):
                s = self.clusters[_]
                for idx in s:
                    self.is_clustered[idx]=_
                    self.unlabled[idx]=0

            self.unlabled=self.unlabled.tolist()

            self.unlabled_set=X[self.unlabled]

            for x_index in self.unlabled_set:
                distances = np.array([np.linalg.norm(X[x_index] - c[centroid]) for centroid in c])
                r=np.argmin(distances)
                self.clusters[r].add(x_index)
                self.is_clustered[x_index]=r

            previous = dict(c)

            for _center in self.clusters:
                lst = []
                for index_value in self.clusters[_center]:
                    lst.append(X[index_value])
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

    def violate_constraints(self, data_index, cluster_index, ml, cl):
        for i in ml[data_index]:
            if self.is_clustered[i] != -1 and self.is_clustered[i] != cluster_index:
                return True

        for i in cl[data_index]:
            if self.is_clustered[i] != -1 and self.is_clustered[i] == cluster_index:
                return True
        return False

    def predict(self, X=None, Transductive=True):
        if Transductive:
            result = self.y
        else:
            result = np.array([])
            for _ in range(len(X)):
                distances = np.array([np.linalg.norm(X[_] - self.center[centroid]) for centroid in self.center])
                result = np.hstack([result, np.argmin(distances)])
        return result
