from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
import numpy as np
from sklearn.semi_supervised._self_training import SelfTrainingClassifier
from sklearn.base import ClassifierMixin
class Self_training(InductiveEstimator,ClassifierMixin):
    def __init__(self,base_estimator,
                threshold=0.75,
                criterion="threshold",
                k_best=10,
                max_iter=10,
                verbose=False,):
        self.base_estimator=base_estimator
        self.threshold=threshold
        self.criterion=criterion
        self.k_best=k_best
        self.max_iter=max_iter
        self.verbose=verbose
        self.self_training=SelfTrainingClassifier(base_estimator=self.base_estimator,
                                                  threshold=self.threshold,
                                                  criterion=self.criterion,
                                                  k_best=self.k_best,
                                                  max_iter=self.max_iter,
                                                  verbose=self.verbose)
        self._estimator_type=ClassifierMixin._estimator_type

    def fit(self, X, y,unlabeled_X):
        U=len(unlabeled_X)
        N = len(X) + len(unlabeled_X)
        _X = np.vstack([X, unlabeled_X])
        unlabeled_y = np.ones(U)*-1
        _y = np.hstack([y, unlabeled_y])
        # print(len(_X))
        # print(len(_y))
        self.self_training=self.self_training.fit(_X,_y)
        return self

    def predict(self,X):
        return self.self_training.predict(X)

    def predict_proba(self,X):
        return self.self_training.predict_proba(X)

    def predict_log_proba(self,X):
        return self.self_training.predict_log_proba(X)

    def decision_function(self, X):
        return self.self_training.decision_function(X)

    def score(self, X, y, sample_weight=None):
        return self.self_training.score(X,y)
