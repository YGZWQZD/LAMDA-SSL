import copy

from lamda_ssl.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
class S3VM(InductiveEstimator,ClassifierMixin):
    def __init__(
            self,
            Cl=1.0,
            Cu=0.001,
            kernel=rbf_kernel,
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
        self._estimator_type = ClassifierMixin._estimator_type
        self.unlabeled_X=None
        self.unlabeled_y=None
        self.class_dict=None
        self.rev_class_dict=None

    def fit(self,X,y,unlabeled_X):
        Cu=self.Cu
        L=len(X)
        N = len(X) + len(unlabeled_X)
        sample_weight = np.ones(N)
        sample_weight[len(X):] = 1.0*Cu/self.Cl
        classes, y_indices = np.unique(y, return_inverse=True)
        # for i in range(len(y)):
        #     if y[i]==0:
        #         y[i]=-1
        if len(classes)!=2:
            raise ValueError('TSVM can only be used in binary classification.')
        # print(classes)

        self.class_dict={classes[0]:-1,classes[1]:1}
        self.rev_class_dict = {-1:classes[0] ,  1:classes[1]}
        y=copy.copy(y)
        for _ in range(L):
            y[_]=self.class_dict[y[_]]

        self.clf.fit(X, y)

        unlabeled_y = self.clf.predict(unlabeled_X)

        # y = np.expand_dims(copy.copy(y), 1)

        # unlabeled_y = np.expand_dims(unlabeled_y, 1)

        u_X_id = np.arange(len(unlabeled_y))
        _X = np.vstack([X, unlabeled_X])
        _y = np.hstack([y, unlabeled_y])


        while Cu < self.Cl:
            # print(self.Cu)
            self.clf.fit(_X, _y, sample_weight=sample_weight)
            while True:
                unlabeled_y_d = self.clf.decision_function(unlabeled_X)    # linear: w^Tx + b

                epsilon = 1 - unlabeled_y * unlabeled_y_d   # calculate function margin

                positive_set, positive_id = epsilon[unlabeled_y > 0], u_X_id[unlabeled_y > 0]
                negative_set, negative_id = epsilon[unlabeled_y < 0], u_X_id[unlabeled_y < 0]
                positive_max_id = positive_id[np.argmax(positive_set)]
                negative_max_id = negative_id[np.argmax(negative_set)]
                a, b = epsilon[positive_max_id], epsilon[negative_max_id]
                if a > 0 and b > 0 and a + b > 2.0:
                    unlabeled_y[positive_max_id] = unlabeled_y[positive_max_id] * -1
                    unlabeled_y[negative_max_id] = unlabeled_y[negative_max_id] * -1
                    # unlabeled_y = np.expand_dims(unlabeled_y, 1)
                    _y = np.hstack([y, unlabeled_y])
                    self.clf.fit(_X, _y, sample_weight=sample_weight)
                else:
                    break
            Cu = min(2*Cu, self.Cl)
            sample_weight[len(X):] = 1.0 * Cu/self.Cl
        self.unlabeled_X = unlabeled_X
        self.unlabeled_y = unlabeled_y
        return self

    def predict(self,X=None):

        result= self.clf.predict(X)
        _len=len(result)
        result=copy.copy(result)
        for _ in range(_len):
            result[_]=self.rev_class_dict[result[_]]
        return result

    def score(self,X=None, y=None,sample_weight=None,Transductive=True):

        if Transductive:
            return self.clf.score(self.unlabeled_X,self.unlabeled_y,sample_weight)
        else:

            _len=len(X)
            y=copy.copy(y)
            for _ in range(_len):
                y[_] = self.class_dict[y[_]]
            return self.clf.score(X, y,sample_weight)



