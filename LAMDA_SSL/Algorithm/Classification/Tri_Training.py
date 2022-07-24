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
        lb_X, lb_y = [[]] * 3, [[]] * 3
        improve = True
        self.iter = 0
        while improve:
            self.iter += 1
            for i in range(3):
                j, k = np.delete(np.array([0, 1, 2]), i)
                update[i] = False
                e[i] = self.measure_error(X, y, j, k)
                if e[i] < e_prime[i]:
                    ulb_y_j = self.estimators[j].predict(unlabeled_X)
                    ulb_y_k = self.estimators[k].predict(unlabeled_X)
                    lb_X[i] = unlabeled_X[ulb_y_j == ulb_y_k]
                    lb_y[i] = ulb_y_j[ulb_y_j == ulb_y_k]
                    if l_prime[i] == 0:
                        l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)
                    if l_prime[i] < len(lb_y[i]):
                        if e[i] * len(lb_y[i]) < e_prime[i] * l_prime[i]:
                            update[i] = True
                        elif l_prime[i] > e[i] / (e_prime[i] - e[i]):
                            lb_index = np.random.choice(len(lb_y[i]), int(e_prime[i] * l_prime[i] / e[i] - 1))
                            lb_X[i], lb_y[i] = lb_X[i][lb_index], lb_y[i][lb_index]
                            update[i] = True
            for i in range(3):
                if update[i]:
                    self.estimators[i].fit(np.append(X, lb_X[i], axis=0), np.append(y, lb_y[i], axis=0))
                    e_prime[i] = e[i]
                    l_prime[i] = len(lb_y[i])
            if update == [False] * 3:
                improve = False
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