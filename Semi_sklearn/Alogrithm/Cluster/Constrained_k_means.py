import numpy as np
from sklearn.base import ClusterMixin
from Semi_sklearn.Base.TransductiveEstimator import TransductiveEstimator
import random
class Constrained_k_means(TransductiveEstimator,ClusterMixin):
    def __init__(self,k, tolerance=1e-7, max_iterations=300):
        self.k=k
        self.tolerance=tolerance
        self.max_iterations=max_iterations
        self.y=None

    def fit(self,X,y=None,unlabeled_X=None,cl=None,ml=None):

        index_list = list(range(len(X)))
        random.shuffle(index_list)
        c= X[index_list[:self.k]]

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

        for _ in range(self.max_iterations):

            self.is_clustered = np.array([-1]*len(X))

            self.clusters = {}
            for i in range(self.k):
                self.clusters[i] = set()

            for x_index in range(len(X)):

                # print(self.ml[x_index])
                # print(self.cl[x_index])
                distances = {center_index: np.linalg.norm(X[x_index] - c[center_index])for center_index in range(len(c))}

                sorted_distances = sorted(distances.items(), key=lambda kv: kv[1])
                # print(sorted_distances)
                empty_flag = True

                for center_index,dis_value in sorted_distances:
                    # print(center_index,dis_value)
                    vc_result = self.violate_constraints(x_index, center_index, self.ml, self.cl)

                    if not vc_result:
                        self.clusters[center_index].add(x_index)
                        self.is_clustered[x_index] = center_index
                        for j in self.ml[x_index]:
                            self.is_clustered[j] = center_index
                        empty_flag = False
                        break

                if empty_flag:
                    return "Clustering Not Found Exception"

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

        self.center=c
        self.y=self.is_clustered
        # print(self.y)
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
