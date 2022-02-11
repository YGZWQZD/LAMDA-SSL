import numpy as np
from sklearn.base import ClusterMixin
from Semi_sklearn.Base.TransductiveEstimator import TransductiveEstimator
import random
class Constrained_k_means(TransductiveEstimator,ClusterMixin):
    def __init__(self,k, tolerance=0.00001, max_iterations=300):
        self.k=k
        self.tolerance=tolerance
        self.max_iterations=max_iterations

    def fit(self,X,y=None,unlabled_X=None,cl=None,ml=None):
        index_list = list(range(len(X)))
        random.shuffle(index_list)
        c= X[index_list[:self.k]]

        for i in range(self.max_iterations):

            self.is_clustered = np.ones(len(X))*-1

            self.clusters = {}
            for i in range(self.k):
                self.clusters[i] = set()

            for x_index in range(len(X)):
                distances = {center_index: np.linalg.norm(X[x_index] - c[center_index])for center_index in range(len(c))}

                sorted_distances = sorted(distances.items(), key=lambda kv: kv[1])

                empty_flag = True

                for center_index,dis_value in sorted_distances:
                    vc_result = self.violate_constraints(x_index, center_index, ml, cl)
                    if not vc_result:
                        self.clusters[center_index].add(x_index)
                        self.is_clustered[x_index] = center_index

                        for j in ml[x_index]:
                            self.is_clustered[j] = center_index

                        empty_flag = False
                        break
                if empty_flag:
                    return "Clustering Not Found Exception"
            previous = dict(c)

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
                if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
                    isOptimal = False
            if isOptimal:
                break
        self.center=c
        self.y=self.is_clustered

    def violate_constraints(self, data_index, cluster_index, ml, cl):
        for i in ml[data_index]:
            if self.is_clustered[i] != -1 and self.is_clustered[i] != cluster_index:
                return True

        for i in cl[data_index]:
            if self.is_clustered[i] != -1 and self.is_clustered[i] == cluster_index:
                return True
        return False

    def predict(self, X=None,Transductive=True):
        if Transductive:
            result=self.y
        else:
            result=np.array([])
            for _ in range(len(X)):
                distances = np.array([np.linalg.norm(X[_] - self.center[centroid]) for centroid in self.center])
                result = np.hstack([result,np.array([np.argmin(distances)])])
        return  result
