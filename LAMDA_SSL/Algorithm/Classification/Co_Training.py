from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
import numpy as np
from sklearn.base import ClassifierMixin
import random
import copy
from torch.utils.data.dataset import Dataset
import LAMDA_SSL.Config.Co_Training as config

class Co_Training(InductiveEstimator,ClassifierMixin):
    # Binary
    def __init__(self, base_estimator=config.base_estimator, base_estimator_2=config.base_estimator_2,
                 p=config.p, n=config.n, k=config.k, s=config.s, evaluation=config.evaluation,
                 verbose=config.verbose, file=config.file):
        # >> Parameter:
        # >> - base_estimator: the first learner for co-training.
        # >> - base_estimator_2: the second learner for co-training.
        # >> - p: In each round, each base learner selects at most p positive samples to assign pseudo-labels.
        # >> - n: In each round, each base learner selects at most n negative samples to assign pseudo-labels.
        # >> - k: iteration rounds.
        # >> - s: the size of the buffer pool in each iteration.

        self.base_estimator = base_estimator
        self.base_estimator_2=base_estimator_2
        self.p=p
        self.n=n
        self.k=k
        self.s=s
        self.evaluation = evaluation
        self.verbose = verbose
        self.file = file
        if isinstance(self.base_estimator,(list,tuple)):
            self.base_estimator,self.base_estimator_2=self.base_estimator[0],self.base_estimator[1]
        if self.base_estimator_2 is None:
            self.base_estimator_2=copy.deepcopy(self.base_estimator)
        self.y_pred=None
        self.y_score=None
        random.seed()
        self._estimator_type = ClassifierMixin._estimator_type

    def fit(self, X, y, unlabeled_X,X_2=None,unlabeled_X_2=None):

        if isinstance(X,(list,tuple)):
            X,X_2=X[0],X[1]

        if isinstance(unlabeled_X,(list,tuple)):
            unlabeled_X,unlabeled_X_2=unlabeled_X[0],unlabeled_X[1]

        # unlabeled_X=copy.copy(unlabeled_X)
        X=copy.copy(X)
        X_2=copy.copy(X_2)
        y=copy.copy(y)

        unlabeled_y=np.ones(len(unlabeled_X))*-1

        u_idx=np.arange(len(unlabeled_X))

        random.shuffle(u_idx)

        pool = u_idx[-min(len(u_idx), self.s):]

        u_idx=u_idx[:-len(pool)]

        it = 0  # number of cotraining iterations we've done so far

        while it != self.k and len(u_idx):
            it += 1

            self.base_estimator.fit(X, y)
            self.base_estimator_2.fit(X_2, y)

            y1_prob = self.base_estimator.predict_proba(unlabeled_X[pool])
            y2_prob = self.base_estimator_2.predict_proba(unlabeled_X_2[pool])

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
            unlabeled_y[[pool[x] for x in p_list]] = 1
            unlabeled_y[[pool[x] for x in n_list]] = 0

            for x in p_list:
                X = np.vstack([X, unlabeled_X[pool[x]]])
                X_2 = np.vstack([X_2, unlabeled_X_2[pool[x]]])
                y=np.hstack([y,unlabeled_y[pool[x]]])

            for x in n_list:
                X = np.vstack([X, unlabeled_X[pool[x]]])
                X_2 = np.vstack([X_2, unlabeled_X_2[pool[x]]])
                y=np.hstack([y,unlabeled_y[pool[x]]])


            pool = [elem for elem in pool if not (elem in p_list or elem in n_list)]

            # add new elements to U_
            add_counter = 0  # number we have added from U to U_
            num_to_add = len(p_list) + len(n_list)
            while add_counter != num_to_add and len(u_idx):
                add_counter += 1
                pool.append(u_idx[0])
                u_idx=u_idx[1:]

        self.base_estimator.fit(X, y)
        self.base_estimator_2.fit(X_2, y)

    def supports_proba(self, clf, x):
        try:
            clf.predict_proba([x])
            return True
        except:
            return False

    def predict(self, X, X_2=None):
        y_proba=self.predict_proba(X=X,X_2=X_2)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

    def predict_proba(self,  X,X_2=None):
        if isinstance(X,(list,tuple)):
            X,X_2=X[0],X[1]
        y_proba = np.full((X.shape[0], 2), -1, np.float)

        y1_proba = self.base_estimator.predict_proba(X)
        y2_proba = self.base_estimator_2.predict_proba(X_2)

        for i, (y1_i_dist, y2_i_dist) in enumerate(zip(y1_proba, y2_proba)):
            y_proba[i][0] = (y1_i_dist[0] + y2_i_dist[0]) / 2
            y_proba[i][1] = (y1_i_dist[1] + y2_i_dist[1]) / 2

        _epsilon = 0.0001
        assert all(abs(sum(y_dist) - 1) <= _epsilon for y_dist in y_proba)
        return y_proba

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