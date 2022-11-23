import copy
import numbers
import numpy as np
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
from torch.utils.data.dataset import Dataset
from sklearn.neighbors import KNeighborsClassifier
import LAMDA_SSL.Config.Assemble as config


class Assemble(InductiveEstimator, ClassifierMixin):
    def __init__(
            self,
            base_estimator=config.base_estimator,
            T=config.T,
            alpha=config.alpha,
            beta=config.beta,
            evaluation=config.evaluation,
            verbose=config.verbose,
            file=config.file):
        # >> Parameter:
        # >> - base_estimator: A base learner for ensemble learning.
        # >> - T: the number of base learners. It is also the number of iterations.
        # >> - alpha: the weight of each sample when the sampling distribution is updated.
        # >> - Beta: used to initialize the sampling distribution of labeled data and unlabeled data.
        self.base_estimator = base_estimator
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.KNN = KNeighborsClassifier(n_neighbors=3)
        self.evaluation = evaluation
        self.verbose = verbose
        self.file = file
        self.w = []
        self.f = []
        self.y_pred = None
        self.y_score = None
        self._estimator_type = ClassifierMixin._estimator_type

    def predict_proba(self, X):
        y_proba = 0
        for _ in range(len(self.w)):
            y_proba = y_proba + self.w[_] * self.f[_].predict_proba(X)
        return y_proba

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

    def fit(self, X, y, unlabeled_X):
        self.w = []
        self.f = []
        l = X.shape[0]
        u = unlabeled_X.shape[0]
        sample_weight = np.zeros(l + u)
        for i in range(l):
            sample_weight[i] = self.beta / l
        for i in range(u):
            sample_weight[i + l] = (1 - self.beta) / u
        unlabeled_y = self.KNN.fit(X, y).predict(unlabeled_X)
        classfier = copy.deepcopy(self.base_estimator)
        X_all = np.concatenate((X, unlabeled_X))
        y_all = np.concatenate((y, unlabeled_y))
        classfier.fit(X_all, y_all, sample_weight=sample_weight * (l + u))
        for i in range(self.T):
            self.f.append(classfier)
            _y_all = classfier.predict(X_all)
            epsilon = 0
            for _ in range(l + u):
                if _y_all[_] != y_all[_]:
                    epsilon += sample_weight[_]
            w = np.log((1 - epsilon) / (epsilon + 1e-8)) * 0.5
            self.w.append(w)
            if epsilon > 0.5:
                break
            probas = self.predict_proba(X_all)
            logits = np.max(probas, axis=1)
            unlabeled_y = self.predict(unlabeled_X)
            y_all = np.concatenate((y, unlabeled_y))
            if isinstance(self.alpha, numbers.Number):
                alpha = np.ones(l + u) * self.alpha
            else:
                alpha = self.alpha
            sample_weight = alpha * np.exp(-logits)
            sample_weight = (sample_weight + 1e-8) / \
                (sample_weight + 1e-8).sum()
            idx_sample = np.random.choice(
                l + u, l, False, p=sample_weight.tolist())
            X_sample = X_all[idx_sample]
            y_sample = y_all[idx_sample]
            sample_weight_sample = sample_weight[idx_sample]
            classfier = copy.deepcopy(self.base_estimator)
            classfier.fit(
                X_sample,
                y_sample,
                sample_weight_sample *
                X_sample.shape[0])
        return self

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
