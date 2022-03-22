from Semi_sklearn.Base.TransductiveEstimator import TransductiveEstimator
from sklearn.base import ClassifierMixin
import numpy as np
from sklearn.semi_supervised._label_propagation import LabelPropagation
class Label_propagation(TransductiveEstimator,ClassifierMixin):
    def __init__(
        self,
        kernel="rbf",
        gamma=20,
        n_neighbors=7,
        max_iter=30,
        tol=1e-3,
        n_jobs=None,
    ):

        self.max_iter = max_iter
        self.tol = tol

        # kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors

        # clamping factor

        self.n_jobs = n_jobs

        self.model=LabelPropagation(kernel=self.kernel,gamma=self.gamma,n_neighbors=self.n_neighbors,
                                  max_iter=self.max_iter,tol=self.tol,n_jobs=n_jobs)

        self._estimator_type=ClassifierMixin._estimator_type

    def fit(self,X,y,unlabeled_X=None):
        U=len(unlabeled_X)
        N = len(X) + len(unlabeled_X)
        _X = np.vstack([X, unlabeled_X])
        unlabeled_y = np.ones(U)*-1
        _y = np.hstack([y, unlabeled_y])
        self.model.fit(_X,_y)

        self.unlabeled_X=unlabeled_X
        self.unlabeled_y=self.model.transduction_[-U:]
        self.unlabeled_y_proba=self.model.label_distributions_[-U:]
        return self

    def predict(self,X=None,Transductive=True):
        if Transductive:
            result=self.unlabeled_y
        else:
            result= self.model.predict(X)
        return result

    def predict_proba(self,X=None,Transductive=True):
        if Transductive:
            result=self.unlabeled_y_proba
        else:
            result= self.model.predict_proba(X)
        return result




