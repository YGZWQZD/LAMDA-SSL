from Semi_sklearn.Base.TransductiveEstimator import TransductiveEstimator
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
import numpy as np

class TSVM(TransductiveEstimator,ClassifierMixin):
    def __init__(
            self,
            Cl=1.0,
            Cu=0.001,
            kernel="rbf",
            degree=3,
            gamma="scale",
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape="ovr",
            break_ties=False,
            random_state=None,
    ):
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
        self.clf=SVC(C=self.Cl,
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
        self.unlabled_X=None
        self.unlabled_y=None

    def fit(self,X,y,unlabled_X):
        N = len(X) + len(unlabled_X)
        sample_weight = np.ones(N)
        sample_weight[len(X):] = 1.0*self.Cu/self.Cl
        self.clf.fit(X, y)
        unlabled_y = self.clf.predict(unlabled_X)
        unlabled_y = np.expand_dims(unlabled_y, 1)
        u_X_id = np.arange(len(unlabled_y))
        _X = np.vstack([X, unlabled_X])
        _y = np.vstack([y, unlabled_y])
        while self.Cu < self.Cl:
            self.clf.fit(_X, _y, sample_weight=sample_weight)
            while True:
                unlabled_y_d = self.clf.decision_function(unlabled_X)    # linear: w^Tx + b
                unlabled_y_= unlabled_y.reshape(-1)
                epsilon = 1 - unlabled_y_ * unlabled_y_d   # calculate function margin
                positive_set, positive_id = epsilon[unlabled_y > 0], u_X_id[unlabled_y > 0]
                negative_set, negative_id = epsilon[unlabled_y < 0], u_X_id[unlabled_y < 0]
                positive_max_id = positive_id[np.argmax(positive_set)]
                negative_max_id = negative_id[np.argmax(negative_set)]
                a, b = epsilon[positive_max_id], epsilon[negative_max_id]
                if a > 0 and b > 0 and a + b > 2.0:
                    unlabled_y[positive_max_id] = unlabled_y[positive_max_id] * -1
                    unlabled_y[negative_max_id] = unlabled_y[negative_max_id] * -1
                    unlabled_y = np.expand_dims(unlabled_y, 1)
                    _y = np.vstack([y, unlabled_y])
                    self.clf.fit(_X, _y, sample_weight=sample_weight)
                else:
                    break
            self.Cu = min(2*self.Cu, self.Cl)
            sample_weight[len(X):] = 1.0*self.Cu/self.Cl
            self.unlabled_X = unlabled_X
            self.unlabled_y=unlabled_y

    def predict(self,X=None,Transductive=True):
        if Transductive:
            return self.unlabled_y
        else:
            return self.clf.predict(X)

    def score(self,X=None, y=None,sample_weight=None,Transductive=True):
        if Transductive:
            return self.clf.score(self.unlabled_X,self.unlabled_y,sample_weight)
        else:
            return self.clf.score(X, y,sample_weight)



