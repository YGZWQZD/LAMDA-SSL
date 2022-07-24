from LAMDA_SSL.Base.TransductiveEstimator import TransductiveEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import sklearn.semi_supervised._label_propagation as sklp
from torch.utils.data.dataset import Dataset
import LAMDA_SSL.Config.LabelPropagation as config

class LabelPropagation(TransductiveEstimator,ClassifierMixin):
    def __init__(
        self,
        kernel=config.kernel,
        gamma=config.gamma,
        n_neighbors=config.n_neighbors,
        max_iter=config.max_iter,
        tol=config.tol,
        n_jobs=config.n_jobs,evaluation=config.evaluation,
        verbose=config.verbose,file=config.file
    ):
        # >> Parameter:
        # >> - kernel: The kernel function which can be inputted as a string 'rbf' or 'knn' or as a callable function.
        # >> - gamma: The gamma value when the kernel function is rbf kernel.
        # >> - n_neighbors: The n value when the kernel function is n_neighbors kernel.
        # >> - max_iter: The maximum number of iterations.
        # >> - tol: Convergence tolerance.
        # >> - n_jobs: The number of parallel jobs.
        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors

        self.n_jobs = n_jobs

        self.model=sklp.LabelPropagation(kernel=self.kernel,gamma=self.gamma,n_neighbors=self.n_neighbors,
                                  max_iter=self.max_iter,tol=self.tol,n_jobs=n_jobs)
        self.evaluation = evaluation
        self.verbose=verbose
        self.file=file

        self.y_pred=None
        self.y_score=None
        self._estimator_type=ClassifierMixin._estimator_type

    def fit(self,X,y,unlabeled_X=None):
        U=len(unlabeled_X)
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
            y_pred=self.unlabeled_y
        else:
            y_proba= self.predict_proba(X,Transductive=Transductive)
            y_pred=np.argmax(y_proba, axis=1)
        return y_pred

    def predict_proba(self,X=None,Transductive=True):
        if Transductive:
            y_proba=self.unlabeled_y_proba
        else:
            y_proba= self.model.predict_proba(X)
        return y_proba


    def evaluate(self,X=None,y=None,Transductive=True):

        if isinstance(X,Dataset) and y is None:
            y=getattr(X,'y')

        self.y_score = self.predict_proba(X,Transductive=Transductive)
        self.y_pred=self.predict(X,Transductive=Transductive)


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

