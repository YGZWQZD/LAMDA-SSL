from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
import numpy as np
from sklearn.base import ClassifierMixin
import random
import copy
from torch.utils.data.dataset import Dataset
import LAMDA_SSL.Config.Co_Training as config
from LAMDA_SSL.Split.ViewSplit import ViewSplit


class Co_Training(InductiveEstimator, ClassifierMixin):
    # Binary
    def __init__(
            self,
            base_estimator=config.base_estimator,
            base_estimator_2=config.base_estimator_2,
            p=config.p,
            n=config.n,
            k=config.k,
            s=config.s,
            random_state=config.random_state,
            evaluation=config.evaluation,
            verbose=config.verbose,
            binary=config.binary,
            threshold=config.threshold,
            file=config.file):
        # >> Parameter:
        # >> - base_estimator: the first learner for co-training.
        # >> - base_estimator_2: the second learner for co-training.
        # >> - p: In each round, each base learner selects at most p positive samples to assign pseudo-labels.
        # >> - n: In each round, each base learner selects at most n negative samples to assign pseudo-labels.
        # >> - k: iteration rounds.
        # >> - s: the size of the buffer pool in each iteration.
        self.base_estimator = base_estimator
        self.base_estimator_2 = base_estimator_2
        self.p = p
        self.n = n
        self.k = k
        self.s = s
        self.random_state = random_state
        self.evaluation = evaluation
        self.binary = binary
        self.threshold = threshold
        self.verbose = verbose
        self.file = file
        if isinstance(self.base_estimator, (list, tuple)):
            self.base_estimator, self.base_estimator_2 = self.base_estimator[
                0], self.base_estimator[1]
        if self.base_estimator_2 is None:
            self.base_estimator_2 = copy.deepcopy(self.base_estimator)
        self.y_pred = None
        self.y_score = None
        random.seed(self.random_state)
        self._estimator_type = ClassifierMixin._estimator_type

    def fit(self, X, y, unlabeled_X, X_2=None, unlabeled_X_2=None):
        if X_2 is None:
            if isinstance(X, (list, tuple)):
                X, X_2 = X[0], X[1]
            else:
                X, X_2 = ViewSplit(X, shuffle=False)
        if unlabeled_X_2 is None:
            if isinstance(unlabeled_X, (list, tuple)):
                unlabeled_X, unlabeled_X_2 = unlabeled_X[0], unlabeled_X[1]
            else:
                unlabeled_X, unlabeled_X_2 = ViewSplit(
                    unlabeled_X, shuffle=False)

        X = copy.copy(X)
        X_2 = copy.copy(X_2)
        y = copy.copy(y)

        unlabeled_y = np.ones(len(unlabeled_X)) * -1

        unlabeled_idx = np.arange(len(unlabeled_X))

        random.shuffle(unlabeled_idx)

        selected_unlabeled_idx = unlabeled_idx[-min(
            len(unlabeled_idx), self.s):]

        unlabeled_idx = unlabeled_idx[:-len(selected_unlabeled_idx)]

        it = 0

        while it != self.k and len(unlabeled_idx):
            it += 1

            self.base_estimator.fit(X, y)
            self.base_estimator_2.fit(X_2, y)

            proba_1 = self.base_estimator.predict_proba(
                unlabeled_X[selected_unlabeled_idx])
            proba_2 = self.base_estimator_2.predict_proba(
                unlabeled_X_2[selected_unlabeled_idx])
            if self.binary:
                negative_samples, positive_samples = [], []

                for i in (proba_1[:, 0].argsort())[-self.n:]:
                    if proba_1[i, 0] > self.threshold:
                        negative_samples.append(i)
                for i in (proba_1[:, 1].argsort())[-self.p:]:
                    if proba_1[i, 1] > self.threshold:
                        positive_samples.append(i)

                for i in (proba_2[:, 0].argsort())[-self.n:]:
                    if proba_2[i, 0] > self.threshold:
                        negative_samples.append(i)
                for i in (proba_2[:, 1].argsort())[-self.p:]:
                    if proba_2[i, 1] > self.threshold:
                        positive_samples.append(i)
                unlabeled_y[[selected_unlabeled_idx[x]
                             for x in positive_samples]] = 1
                unlabeled_y[[selected_unlabeled_idx[x]
                             for x in negative_samples]] = 0

                for x in positive_samples:
                    X = np.vstack([X, unlabeled_X[selected_unlabeled_idx[x]]])
                    X_2 = np.vstack(
                        [X_2, unlabeled_X_2[selected_unlabeled_idx[x]]])
                    y = np.hstack([y, unlabeled_y[selected_unlabeled_idx[x]]])

                for x in negative_samples:
                    X = np.vstack([X, unlabeled_X[selected_unlabeled_idx[x]]])
                    X_2 = np.vstack(
                        [X_2, unlabeled_X_2[selected_unlabeled_idx[x]]])
                    y = np.hstack([y, unlabeled_y[selected_unlabeled_idx[x]]])

                selected_unlabeled_idx = np.array([elem for elem in selected_unlabeled_idx if not (
                    elem in positive_samples or elem in negative_samples)])
                num_selected = len(positive_samples) + len(negative_samples)

            else:
                pred_1 = np.argmax(proba_1, axis=1)
                pred_2 = np.argmax(proba_2, axis=1)
                confidence_1 = np.max(proba_1, axis=1)
                confidence_2 = np.max(proba_2, axis=1)
                selected_1 = confidence_1 > self.threshold
                selected_2 = confidence_2 > self.threshold
                unlabeled_y[selected_unlabeled_idx[selected_1]] = pred_1[selected_1]
                unlabeled_y[selected_unlabeled_idx[selected_2]] = pred_2[selected_2]
                selected_samples=[]
                for i in (confidence_1.argsort())[-self.n:]:
                    if confidence_1[i] > self.threshold:
                        selected_samples.append(i)
                for i in (confidence_2.argsort())[-self.n:]:
                    if confidence_2[i] > self.threshold:
                        selected_samples.append(i)
                for x in selected_samples:
                    X = np.vstack([X, unlabeled_X[selected_unlabeled_idx[x]]])
                    X_2 = np.vstack(
                        [X_2, unlabeled_X_2[selected_unlabeled_idx[x]]])
                    y = np.hstack([y, unlabeled_y[selected_unlabeled_idx[x]]])
                selected_unlabeled_idx = np.array(
                    [elem for elem in selected_unlabeled_idx if not (elem in selected_samples)])
                num_selected = len(selected_samples)
            num_selected = min(num_selected, len(unlabeled_idx))
            selected_unlabeled_idx = np.concatenate(
                (selected_unlabeled_idx, unlabeled_idx[:num_selected]))
            unlabeled_idx = unlabeled_idx[num_selected:]

        self.base_estimator.fit(X, y)
        self.base_estimator_2.fit(X_2, y)
        return self

    def predict(self, X, X_2=None):
        y_proba = self.predict_proba(X=X, X_2=X_2)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

    def predict_proba(self, X, X_2=None):
        if X_2 is None:
            if isinstance(X, (list, tuple)):
                X, X_2 = X[0], X[1]
            else:
                X, X_2 = ViewSplit(X, shuffle=False)
        # y_proba = np.full((X.shape[0], 2), -1, np.float)

        y1_proba = self.base_estimator.predict_proba(X)
        y2_proba = self.base_estimator_2.predict_proba(X_2)

        # for i, (y1_i_dist, y2_i_dist) in enumerate(zip(y1_proba, y2_proba)):
        #     y_proba[i][0] = (y1_i_dist[0] + y2_i_dist[0]) / 2
        #     y_proba[i][1] = (y1_i_dist[1] + y2_i_dist[1]) / 2
        y_proba=(y1_proba+y2_proba)/2
        return y_proba

    def evaluate(self, X, y=None):

        if isinstance(X, Dataset) and y is None:
            y = getattr(X, 'y')

        self.y_score = self.predict_proba(X)
        self.y_pred = self.predict(X)

        if self.evaluation is None:
            return None
        elif isinstance(self.evaluation, (list, tuple)):
            performance = []
            for eval in self.evaluation:
                score = eval.scoring(y, self.y_pred, self.y_score)
                if self.verbose:
                    print(score, file=self.file)
                performance.append(score)
            self.performance = performance
            return performance
        elif isinstance(self.evaluation, dict):
            performance = {}
            for key, val in self.evaluation.items():

                performance[key] = val.scoring(y, self.y_pred, self.y_score)

                if self.verbose:
                    print(key, ' ', performance[key], file=self.file)
                self.performance = performance
            return performance
        else:
            performance = self.evaluation.scoring(y, self.y_pred, self.y_score)
            if self.verbose:
                print(performance, file=self.file)
            self.performance = performance
            return performance
