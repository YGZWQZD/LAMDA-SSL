import copy
from LAMDA_SSL.Base.TransductiveEstimator import TransductiveEstimator
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
import numpy as np
from torch.utils.data.dataset import Dataset
import LAMDA_SSL.Config.TSVM as config

class TSVM(TransductiveEstimator,ClassifierMixin):
    def __init__(
            self,
            Cl=config.Cl,
            Cu=config.Cu,
            kernel=config.kernel,
            degree=config.degree,
            gamma=config.gamma,
            coef0=config.coef0,
            shrinking=config.shrinking,
            probability=config.probability,
            tol=config.tol,
            cache_size=config.cache_size,
            class_weight=config.class_weight,
            max_iter=config.max_iter,
            decision_function_shape=config.decision_function_shape,
            break_ties=config.break_ties,
            random_state=config.random_state,evaluation=config.evaluation,
            verbose=config.verbose,file=config.file
    ):
        # >> Parameter:
        # >> - Cl: The weight of labeled samples.
        # >> - Cu: The weight of unlabeled samples.
        # >> - kernel: 'rbf'、'knn' or callable. Specifies the kernel type to be used in the algorithm.
        # >> - degree: The polynomial order corresponding to the 'poly' kernel.
        # >> - gamma: The gamma parameter corresponding to the kernel. It is valid when kernel is 'rbf', 'poly' or 'sigmoid'.
        # >> - coef0: The constant term of the kernel function. It is valid when kernel is 'poly' or 'sigmoid'.
        # >> - shrinking: Whether to use the shrinking heuristic method.
        # >> - probability: Weights for rotation angle classification loss.
        # >> - tol: Tolerance to stop training, default is 1e-3.
        # >> - cache_size: The cache size of the Kernel function.
        # >> - class_weight: The weights of different classes.
        # >> - verbose: Whether to allow redundant output.
        # >> - max_iter: The maximum number of iterations. -1 for unlimited.
        # >> - decision_function_shape: {'ovo', 'ovr'}, default='ovr'. Whether to return a one-vs-rest ('ovr') decision function of shape(n_samples, n_classes) as all other classifiers, or the original one-vs-one ('ovo') decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one ('ovo') is always used as multi-class strategy. The parameter is ignored for binary classification.
        # >> - break_ties: Whether to classify by calculating confidence in the event of a tie.
        # >> - random_state: A random seed for data shuffling.
        self.Cl = Cl
        self.Cu = Cu
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state
        self.base_estimator=SVC(C=self.Cl,
                    kernel=self.kernel,
                    degree = self.degree,
                    gamma = self.gamma,
                    coef0 = self.coef0,
                    shrinking = self.shrinking,
                    probability = self.probability,
                    tol = self.tol,
                    cache_size = self.cache_size,
                    class_weight = self.class_weight,
                    verbose = self.verbose,
                    max_iter = self.max_iter,
                    decision_function_shape = self.decision_function_shape,
                    break_ties = self.break_ties,
                    random_state = self.random_state)
        self.unlabeled_X=None
        self.unlabeled_y=None
        self.class_dict=None
        self.rev_class_dict=None
        self.unlabeled_y_d=None
        self.evaluation = evaluation
        self.verbose = verbose
        self.file = file
        self.y_pred=None
        self.y_score=None
        self._estimator_type = ClassifierMixin._estimator_type

    def fit(self,X,y,unlabeled_X):
        L=len(X)
        N = len(X) + len(unlabeled_X)
        sample_weight = np.ones(N)
        sample_weight[len(X):] = 1.0*self.Cu/self.Cl
        classes, y_indices = np.unique(y, return_inverse=True)
        if len(classes)!=2:
            raise ValueError('TSVM can only be used in binary classification.')

        self.class_dict={classes[0]:-1,classes[1]:1}
        self.rev_class_dict = {-1:classes[0] ,  1:classes[1]}
        y=copy.copy(y)
        for _ in range(L):
            y[_]=self.class_dict[y[_]]

        self.base_estimator.fit(X, y)

        unlabeled_y = self.base_estimator.predict(unlabeled_X)

        u_X_id = np.arange(len(unlabeled_y))
        _X = np.vstack([X, unlabeled_X])
        _y = np.hstack([y, unlabeled_y])


        while self.Cu < self.Cl:
            self.base_estimator.fit(_X, _y, sample_weight=sample_weight)
            while True:
                unlabeled_y_d = self.base_estimator.decision_function(unlabeled_X)
                self.unlabeled_y_d=unlabeled_y_d
                epsilon = 1 - unlabeled_y * unlabeled_y_d

                positive_set, positive_id = epsilon[unlabeled_y > 0], u_X_id[unlabeled_y > 0]
                negative_set, negative_id = epsilon[unlabeled_y < 0], u_X_id[unlabeled_y < 0]
                positive_max_id = positive_id[np.argmax(positive_set)]
                negative_max_id = negative_id[np.argmax(negative_set)]
                a, b = epsilon[positive_max_id], epsilon[negative_max_id]
                if a > 0 and b > 0 and a + b > 2.0:
                    unlabeled_y[positive_max_id] = unlabeled_y[positive_max_id] * -1
                    unlabeled_y[negative_max_id] = unlabeled_y[negative_max_id] * -1
                    _y = np.hstack([y, unlabeled_y])
                    self.base_estimator.fit(_X, _y, sample_weight=sample_weight)
                else:
                    break
            self.Cu = min(2*self.Cu, self.Cl)
            sample_weight[len(X):] = 1.0*self.Cu/self.Cl
        self.unlabeled_X = unlabeled_X
        self.unlabeled_y = unlabeled_y
        return self

    def predict_proba(self, X=None, Transductive=True):
        if Transductive:
            y_proba = np.full((self.unlabeled_y_d.shape[0], 2), 0, np.float)
            y_proba[:, 0] = 1 / (1 + np.exp(self.unlabeled_y_d))
            y_proba[:, 1] = 1 - y_proba[:, 0]
        else:
            y_proba= self.base_estimator.predict_proba(X)
        return y_proba

    def predict(self,X=None,Transductive=True):
        if Transductive:
            result=self.unlabeled_y
        else:
            result= self.base_estimator.predict(X)
        _len=len(result)
        result=copy.copy(result)
        for _ in range(_len):
            result[_]=self.rev_class_dict[result[_]]
        return result

    def score(self,X=None, y=None,sample_weight=None,Transductive=True):
        if Transductive:
            return self.base_estimator.score(self.unlabeled_X,self.unlabeled_y,sample_weight)
        else:
            _len=len(X)
            y=copy.copy(y)
            for _ in range(_len):
                y[_] = self.class_dict[y[_]]
            return self.base_estimator.score(X, y,sample_weight)

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