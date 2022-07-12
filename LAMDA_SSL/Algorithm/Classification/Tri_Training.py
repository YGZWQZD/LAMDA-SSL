import copy
import numpy as np
import sklearn
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
from torch.utils.data.dataset import Dataset
import  LAMDA_SSL.Config.Tri_Training as config

class Tri_Training(InductiveEstimator,ClassifierMixin):
    def __init__(self, base_estimator=config.base_estimator,base_estimator_2=config.base_estimator_2,
                 base_estimator_3=config.base_estimator_3,evaluation=config.evaluation,
                 verbose=config.verbose,file=config.file):
        # >> Parameter:
        # >> - base_estimator: The first base learner in TriTraining.
        # >> - base_estimator_2: The second base learner in TriTraining.
        # >> - base_estimator_3: The third base learner in TriTraining.
        if isinstance(base_estimator,(list,tuple)):
            self.estimators=base_estimator
        else:
            self.estimators=[base_estimator,base_estimator_2,base_estimator_3]
        if self.estimators[1] is None:
            self.estimators[1]=copy.deepcopy(self.estimators[0])
        if self.estimators[2] is None:
            self.estimators[2] = copy.deepcopy(self.estimators[0])
        self.evaluation = evaluation
        self.verbose=verbose
        self.file=file
        self.y_pred=None
        self.y_score=None
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

    def predict_proba(self,X):
        y_proba = np.full((X.shape[0], 2), 0, np.float)
        for i in range(3):
            y_proba+=self.estimators[i].predict_proba(X)
        return y_proba

    def predict(self, X):
        pred = np.asarray([self.estimators[i].predict(X) for i in range(3)])
        pred[0][pred[1] == pred[2]] = pred[1][pred[1] == pred[2]]
        y_pred=pred[0]
        return y_pred



    def measure_error(self, X, y, j, k):
        j_pred = self.estimators[j].predict(X)
        k_pred = self.estimators[k].predict(X)
        wrong_index = np.logical_and(j_pred != y, k_pred == j_pred)
        # wrong_index =np.logical_and(j_pred != y_test, k_pred!=y_test)
        return sum(wrong_index) / sum(j_pred == k_pred)

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