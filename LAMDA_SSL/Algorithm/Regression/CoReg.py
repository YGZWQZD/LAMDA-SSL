import copy
from sklearn.base import RegressorMixin
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
import numpy as np
from torch.utils.data.dataset import Dataset
import LAMDA_SSL.Config.CoReg as config

class CoReg(InductiveEstimator,RegressorMixin):
    def __init__(self, k1=config.k1, k2=config.k2, p1=config.p1, p2=config.p2,
                 max_iters=config.max_iters, pool_size=config.pool_size,
                 evaluation=config.evaluation,verbose=config.verbose,file=config.file):
        # >> Parameter
        # >> - k1: The k value for the k-nearest neighbors in the first base learner.
        # >> - k2: The k value for the k-nearest neighbors in the second base learner.
        # >> - p1: The order of the distance calculated in the first base learner.
        # >> - p2: The order of the distance calculated in the second base learner.
        # >> - max_iters: The maximum number of iterations.
        # >> - pool_size: The size of the buffer pool.
        self.k1=k1
        self.k2=k2
        self.p1=p1
        self.p2=p2
        self.max_iters=max_iters
        self.pool_size=pool_size
        self.h1 = KNeighborsRegressor(n_neighbors=self.k1, p=self.p1)
        self.h2 = KNeighborsRegressor(n_neighbors=self.k2, p=self.p2)
        self.h1_temp = KNeighborsRegressor(n_neighbors=self.k1, p=self.p1)
        self.h2_temp = KNeighborsRegressor(n_neighbors=self.k2, p=self.p2)
        self.evaluation = evaluation
        self.verbose = verbose
        self.file = file
        self.y_pred=None
        self._estimator_type = RegressorMixin._estimator_type

    def fit(self,X,y,unlabeled_X):
        X1=copy.copy(X)
        X2=copy.copy(X)
        y1=copy.copy(y)
        y2=copy.copy(y)
        unlabeled_X=copy.copy(unlabeled_X)
        self.h1.fit(X1, y1)
        self.h2.fit(X2, y2)

        U_X_pool, U_idx_pool = shuffle(
            unlabeled_X,  range(unlabeled_X.shape[0]))
        U_X_pool = U_X_pool[:self.pool_size]
        U_idx_pool = U_idx_pool[:self.pool_size]

        for _ in range(self.max_iters):
            stop_training = True
            added_idxs = []
            to_remove=[]
            for idx_h in [1, 2]:
                if idx_h == 1:
                    h = self.h1
                    h_temp = self.h1_temp
                    L_X, L_y = X1, y1
                else:
                    h = self.h2
                    h_temp = self.h2_temp
                    L_X, L_y = X2, y2
                deltas = np.zeros((U_X_pool.shape[0],))

                for idx_u, x_u in enumerate(U_X_pool):
                    x_u = x_u.reshape(1, -1)
                    y_u_hat = h.predict(x_u)
                    omega = h.kneighbors(x_u, return_distance=False)[0]
                    X_temp = np.concatenate((L_X, x_u))
                    y_temp = np.concatenate((L_y, y_u_hat))
                    h_temp.fit(X_temp, y_temp)

                    delta = 0
                    for idx_o in omega:
                        delta += (L_y[idx_o].reshape(1, -1) -
                                  h.predict(L_X[idx_o].reshape(1, -1))) ** 2
                        delta -= (L_y[idx_o].reshape(1, -1) -
                                  h_temp.predict(L_X[idx_o].reshape(1, -1))) ** 2

                    deltas[idx_u] = delta
                sort_idxs = np.argsort(deltas)[::-1]  # max to min
                max_idx = sort_idxs[0]
                if max_idx in added_idxs: max_idx = sort_idxs[1]
                if deltas[max_idx] > 0:
                    stop_training = False
                    added_idxs.append(max_idx)
                    x_u = U_X_pool[max_idx].reshape(1, -1)
                    y_u_hat = h.predict(x_u)
                    idx_u=U_idx_pool[max_idx]
                    to_remove.append(idx_u)
                    if idx_h == 1:
                        X1 = np.concatenate((X1, x_u))
                        y1 = np.concatenate((y1, y_u_hat))
                    else:
                        X2 = np.concatenate((X2, x_u))
                        y2 = np.concatenate((y2, y_u_hat))
            if stop_training:
                break
            else:
                self.h1.fit(X1, y1)
                self.h2.fit(X2, y2)
                unlabeled_X = np.delete(unlabeled_X, to_remove, axis=0)
                U_X_pool, U_idx_pool = shuffle(
                    unlabeled_X, range(unlabeled_X.shape[0]))
                U_X_pool = U_X_pool[:self.pool_size]
                U_idx_pool = U_idx_pool[:self.pool_size]
        return self

    def predict(self,X):
        result1 = self.h1.predict(X)
        result2 = self.h2.predict(X)
        result = 0.5 * (result1 + result2)
        return result

    def evaluate(self,X,y=None):

        if isinstance(X,Dataset) and y is None:
            y=getattr(X,'y')

        self.y_pred=self.predict(X)


        if self.evaluation is None:
            return None
        elif isinstance(self.evaluation,(list,tuple)):
            performance=[]
            for eval in self.evaluation:
                score=eval.scoring(y,self.y_pred)
                if self.verbose:
                    print(score, file=self.file)
                performance.append(score)
            self.performance = performance
            return performance
        elif isinstance(self.evaluation,dict):
            performance={}
            for key,val in self.evaluation.items():

                performance[key]=val.scoring(y,self.y_pred)

                if self.verbose:
                    print(key,' ',performance[key],file=self.file)
                self.performance = performance
            return performance
        else:
            performance=self.evaluation.scoring(y,self.y_pred)
            if self.verbose:
                print(performance, file=self.file)
            self.performance=performance
            return performance