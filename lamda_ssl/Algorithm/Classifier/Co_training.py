from lamda_ssl.Base.InductiveEstimator import InductiveEstimator
import numpy as np
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
        self._estimator_type = ClassifierMixin._estimator_type

        if isinstance(self.base_estimator,(list,tuple)):
            self.base_estimator,self.base_estimator_2=self.base_estimator[0],self.base_estimator[1]
        if self.base_estimator_2 is None:
            self.base_estimator_2=copy.deepcopy(self.base_estimator)

        random.seed()

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

            # label the samples and remove the newly added samples from U_

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
        """Checks if a given classifier supports the 'predict_proba' method, given a single vector x"""
        try:
            clf.predict_proba([x])
            return True
        except:
            return False

    def predict(self, X, X_2=None):

        if isinstance(X,(list,tuple)):
            X,X_2=X[0],X[1]

        y1 = self.base_estimator.predict(X)
        y2 = self.base_estimator_2.predict(X_2)

        proba_supported = self.supports_proba(self.base_estimator, X[0]) and self.supports_proba(self.base_estimator, X_2[0])

        # fill y_pred with -1 so we can identify the samples in which the classifiers failed to agree
        y_pred = np.asarray([-1] * X.shape[0])

        for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
            if y1_i == y2_i:
                y_pred[i] = y1_i
            elif proba_supported:
                y1_probs = self.base_estimator.predict_proba([X[i]])[0]
                y2_probs = self.base_estimator_2.predict_proba([X_2[i]])[0]
                sum_y_probs = [prob1 + prob2 for (prob1, prob2) in zip(y1_probs, y2_probs)]
                max_sum_prob = max(sum_y_probs)
                y_pred[i] = sum_y_probs.index(max_sum_prob)

            else:
                # the classifiers disagree and don't support probability, so we guess
                y_pred[i] = random.randint(0, 1)

        # check that we did everything right
        assert not (-1 in y_pred)

        return y_pred

    def predict_proba(self, X1, X2):
        """Predict the probability of the samples belonging to each class."""
        y_proba = np.full((X1.shape[0], 2), -1, np.float)

        y1_proba = self.base_estimator.predict_proba(X1)
        y2_proba = self.base_estimator_2.predict_proba(X2)

        for i, (y1_i_dist, y2_i_dist) in enumerate(zip(y1_proba, y2_proba)):
            y_proba[i][0] = (y1_i_dist[0] + y2_i_dist[0]) / 2
            y_proba[i][1] = (y1_i_dist[1] + y2_i_dist[1]) / 2

        _epsilon = 0.0001
        assert all(abs(sum(y_dist) - 1) <= _epsilon for y_dist in y_proba)
        return y_proba