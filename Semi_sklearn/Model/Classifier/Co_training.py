from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
import numpy as np
from sklearn.semi_supervised._self_training import SelfTrainingClassifier
from sklearn.base import ClassifierMixin
import random
import copy

class Co_training(InductiveEstimator,ClassifierMixin):
    def __init__(self, base_estimator, base_estimator_2=None, p=5, n=5, k=30, s=75):
        self.base_estimator = base_estimator
        self.base_estimator_2=base_estimator_2
        self.p=p
        self.n=n
        self.k=k
        self.s=s

        if isinstance(self.base_estimator,(list,tuple)):
            self.base_estimator,self.base_estimator_2=self.base_estimator[0],self.base_estimator_2[1]
        if self.base_estimator_2 is None:
            self.base_estimator_2=copy.copy(self.base_estimator)

        random.seed()

    def fit(self, X, y, unlabled_X,X_2=None,unlabled_X_2=None):

        if isinstance(X,(list,tuple)):
            X,X_2=X[0],X[1]

        if isinstance(unlabled_X,(list,tuple)):
            unlabled_X,unlabled_X_2=unlabled_X[0],unlabled_X[1]

        # unlabled_X=copy.copy(unlabled_X)
        X=copy.copy(X)
        X_2=copy.copy(X_2)
        y=copy.copy(y)

        unlabled_y=np.ones(unlabled_X)*-1

        u_idx=np.arange(len(unlabled_X))

        random.shuffle(u_idx)

        pool = u_idx[-min(len(u_idx), self.s):]

        u_idx=u_idx[:-len(pool)]

        it = 0  # number of cotraining iterations we've done so far

        while it != self.k and u_idx:
            it += 1

            self.base_estimator.fit(X, y)
            self.base_estimator_2.fit(X_2, y)

            y1_prob = self.base_estimator.predict_proba(unlabled_X)
            y2_prob = self.base_estimator_2.predict_proba(unlabled_X_2)

            n_list, p_list = [], []

            for i in (y1_prob[:, 0].argsort())[-self.n:]:
                if y1_prob[i, 0] > 0.5:
                    n_list.append(i)
            for i in (y1_prob[:, 1].argsort())[-self.p:]:
                if y1_prob[i, 1] > 0.5:
                    p_list.append(i)

            for i in (y2_prob[:, 0].argsort())[-self.n:]:
                if y2_prob[i, 0] > 0.5:
                    n_list.append(i)
            for i in (y2_prob[:, 1].argsort())[-self.p:]:
                if y2_prob[i, 1] > 0.5:
                    p_list.append(i)

            # label the samples and remove thes newly added samples from U_
            unlabled_y[[pool[x] for x in p_list]] = 1
            unlabled_y[[pool[x] for x in n_list]] = 0

            X.extend([unlabled_X[u_idx[x]] for x in p_list])
            X.extend([unlabled_X[u_idx[x]] for x in n_list])

            X_2.extend([unlabled_X_2[u_idx[x]] for x in p_list])
            X_2.extend([unlabled_X_2[u_idx[x]] for x in n_list])

            y.extend([unlabled_y[u_idx[x]] for x in p_list])
            y.extend([unlabled_y[u_idx[x]] for x in n_list])

            pool = [elem for elem in pool if not (elem in p_list or elem in n_list)]

            # add new elements to U_
            add_counter = 0  # number we have added from U to U_
            num_to_add = len(p_list) + len(n_list)
            while add_counter != num_to_add and u_idx:
                add_counter += 1
                pool.append(u_idx.pop())

        self.base_estimator.fit(X, y)
        self.base_estimator_2.fit(X_2, y)
