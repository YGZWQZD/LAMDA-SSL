from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
import numpy as np
from sklearn.semi_supervised._self_training import SelfTrainingClassifier
from sklearn.base import ClassifierMixin
from torch.utils.data.dataset import Dataset
import LAMDA_SSL.Config.Self_Training as config

class Self_Training(InductiveEstimator,ClassifierMixin):
    def __init__(self,base_estimator=config.base_estimator,
                threshold=config.threshold,
                criterion=config.criterion,
                k_best=config.k_best,
                max_iter=config.max_iter,
                evaluation=config.evaluation,verbose=config.verbose,file=config.verbose):
        # >> Parameter:
        # >> - base_estimator: The base supervised learner used in the Self_training algorithm.
        # >> - criterion: There are two forms: 'threshold' and 'k_best', the former selects samples according to the threshold, and the latter selects samples according to the ranking.
        # >> - threshold: When criterion is 'threshold', the threshold used for selecting samples during training.
        # >> - k_best: When criterion is 'k_best', select the top k samples of confidence from training.
        # >> - max_iter: The maximum number of iterations.
        # >> - verbose: Whether to allow redundant output.
        self.base_estimator=base_estimator
        self.threshold=threshold
        self.criterion=criterion
        self.k_best=k_best
        self.max_iter=max_iter
        self.evaluation = evaluation
        self.verbose=verbose
        self.file=file
        self.y_pred=None
        self.y_score=None
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
        self.self_training=self.self_training.fit(_X,_y)
        return self

    def predict(self,X):
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

    def predict_proba(self,X):
        y_proba=self.self_training.predict_proba(X)
        return y_proba

    def predict_log_proba(self,X):
        return self.self_training.predict_log_proba(X)

    def decision_function(self, X):
        return self.self_training.decision_function(X)

    def score(self, X, y, sample_weight=None):
        return self.self_training.score(X,y)

    def evaluate(self,X,y=None):

        if isinstance(X,Dataset) and y is None:
            y=getattr(X,'y')

        self.y_score = self.predict_proba(X)
        self.y_pred=self.predict(X)


        if self.evaluation is None:
            return None
        elif isinstance(self.evaluation,(list,tuple)):
            result=[]
            for eval in self.evaluation:
                score=eval.scoring(y,self.y_pred,self.y_score)
                if self.verbose:
                    print(score, file=self.file)
                result.append(score)
            self.result = result
            return result
        elif isinstance(self.evaluation,dict):
            result={}
            for key,val in self.evaluation.items():

                result[key]=val.scoring(y,self.y_pred,self.y_score)

                if self.verbose:
                    print(key,' ',result[key],file=self.file)
                self.result = result
            return result
        else:
            result=self.evaluation.scoring(y,self.y_pred,self.y_score)
            if self.verbose:
                print(result, file=self.file)
            self.result=result
            return result