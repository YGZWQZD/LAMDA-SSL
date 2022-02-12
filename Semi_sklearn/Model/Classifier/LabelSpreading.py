from Semi_sklearn.Base.TransductiveEstimator import TransductiveEstimator
from sklearn.base import ClassifierMixin
import numpy as np
from sklearn.semi_supervised._label_propagation import LabelSpreading
class Label_spreading(TransductiveEstimator,ClassifierMixin):
    def __init__(
        self,
        kernel="rbf",
        gamma=20,
        n_neighbors=7,
        alpha=0.2,
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
        self.alpha=alpha

        # clamping factor

        self.n_jobs = n_jobs

        self.model=LabelSpreading(kernel=self.kernel,gamma=self.gamma,n_neighbors=self.n_neighbors,
                                  alpha=self.alpha,max_iter=self.max_iter,tol=self.tol,n_jobs=n_jobs)

        self._estimator_type=ClassifierMixin._estimator_type

    def fit(self,X,y,unlabled_X=None):
        U=len(unlabled_X)
        N = len(X) + len(unlabled_X)
        _X = np.vstack([X, unlabled_X])
        unlabled_y = np.ones(U)*-1
        _y = np.hstack([y, unlabled_y])
        self.model.fit(_X,_y)

        self.unlabled_X=unlabled_X
        self.unlabled_y=self.model.transduction_[-U:]
        self.unlabled_y_proba=self.model.label_distributions_[-U:]
        return self

    def predict(self,X=None,Transductive=True):
        if Transductive:
            result=self.unlabled_y
        else:
            result= self.model.predict(X)
        return result

    def predict_proba(self,X=None,Transductive=True):
        if Transductive:
            result=self.unlabled_y_proba
        else:
            result= self.model.predict_proba(X)
        return result




