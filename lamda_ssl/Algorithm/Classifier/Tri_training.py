import copy

import numpy as np
import sklearn
from lamda_ssl.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin

class TriTraining(InductiveEstimator,ClassifierMixin):
    def __init__(self, base_estimator,base_estimator_2=None,base_estimator_3=None):

        if isinstance(base_estimator,(list,tuple)):
            self.estimators=base_estimator
        else:
            self.estimators=[base_estimator,base_estimator_2,base_estimator_3]
        if self.estimators[1] is None:
            self.estimators[1]=copy.deepcopy(self.estimators[0])
        if self.estimators[2] is None:
            self.estimators[2] = copy.deepcopy(self.estimators[0])
        self._estimator_type = ClassifierMixin._estimator_type

    def fit(self, X, y, unlabeled_X):
        for i in range(3):
            sample = sklearn.utils.resample(X, y)
            self.estimators[i].fit(*sample)
        e_prime = [0.5] * 3
        l_prime = [0] * 3
        e = [0] * 3
        update = [False] * 3
        Li_X, Li_y = [[]] * 3, [[]] * 3  # to save proxy labeled data
        improve = True
        self.iter = 0

        while improve:
            self.iter += 1  # count iterations

            for i in range(3):
                j, k = np.delete(np.array([0, 1, 2]), i)
                update[i] = False
                e[i] = self.measure_error(X, y, j, k)
                if e[i] < e_prime[i]:
                    U_y_j = self.estimators[j].predict(unlabeled_X)
                    U_y_k = self.estimators[k].predict(unlabeled_X)
                    Li_X[i] = unlabeled_X[U_y_j == U_y_k]  # when two models agree on the label, save it
                    Li_y[i] = U_y_j[U_y_j == U_y_k]
                    if l_prime[i] == 0:  # no updated before
                        l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)
                    if l_prime[i] < len(Li_y[i]):
                        if e[i] * len(Li_y[i]) < e_prime[i] * l_prime[i]:
                            update[i] = True
                        elif l_prime[i] > e[i] / (e_prime[i] - e[i]):
                            L_index = np.random.choice(len(Li_y[i]), int(e_prime[i] * l_prime[i] / e[i] - 1))
                            Li_X[i], Li_y[i] = Li_X[i][L_index], Li_y[i][L_index]
                            update[i] = True

            for i in range(3):
                if update[i]:
                    self.estimators[i].fit(np.append(X, Li_X[i], axis=0), np.append(y, Li_y[i], axis=0))
                    e_prime[i] = e[i]
                    l_prime[i] = len(Li_y[i])

            if update == [False] * 3:
                improve = False  # if no classifier was updated, no improvement
        return self

    def predict(self, X):
        pred = np.asarray([self.estimators[i].predict(X) for i in range(3)])
        pred[0][pred[1] == pred[2]] = pred[1][pred[1] == pred[2]]
        return pred[0]



    def measure_error(self, X, y, j, k):
        j_pred = self.estimators[j].predict(X)
        k_pred = self.estimators[k].predict(X)
        wrong_index = np.logical_and(j_pred != y, k_pred == j_pred)
        # wrong_index =np.logical_and(j_pred != y_test, k_pred!=y_test)
        return sum(wrong_index) / sum(j_pred == k_pred)