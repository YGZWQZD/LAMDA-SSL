import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import kneighbors_graph
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel,linear_kernel
import copy
from torch.utils.data.dataset import Dataset
import LAMDA_SSL.Config.LapSVM as config
from scipy import sparse

class LapSVM(InductiveEstimator,ClassifierMixin):
    # Binary
    def __init__(self,
           distance_function= config.distance_function,
           gamma_d=config.gamma_d,
           neighbor_mode =config.neighbor_mode,
           t=config.t,
           n_neighbor= config.n_neighbor,
           kernel_function= config.kernel_function,
           gamma_k=config.gamma_k,
           gamma_A= config.gamma_A,
           gamma_I= config.gamma_I,evaluation=config.evaluation,
           verbose=config.verbose,file=config.file):
        # >> Parameter:
        # >> - distance_function: The distance function for building the graph. This Pamater is valid when neighbor_mode is None.
        # >> - gamma_d: Kernel parameters related to distance_function.
        # >> - neighbor_mode: The edge weight after constructing the graph model by k-nearest neighbors. There are two options 'connectivity' and 'distance', 'connectivity' returns a 0-1 matrix, and 'distance' returns a distance matrix.
        # >> - n_neighbor: k value of k-nearest neighbors.
        # >> - kernel_function: The kernel function corresponding to SVM.
        # >> - gamma_k: The gamma parameter corresponding to kernel_function.
        # >> - gamma_A: Penalty weight for function complexity.
        # >> - gamma_I: Penalty weight for smoothness of data distribution.
        self.distance_function=distance_function
        self.neighbor_mode=neighbor_mode
        self.n_neighbor=n_neighbor
        self.t=t
        self.kernel_function=kernel_function
        self.gamma_k=gamma_k
        self.gamma_d=gamma_d
        self.gamma_A=gamma_A
        self.gamma_I=gamma_I
        self.evaluation = evaluation
        self.verbose=verbose
        self.file=file
        self.y_pred=None
        self.y_score=None
        self._estimator_type = ClassifierMixin._estimator_type


    def fit(self,X,y,unlabeled_X):
        classes, y_indices = np.unique(y, return_inverse=True)
        if len(classes)!=2:
            raise ValueError('LapSVM can only be used in binary classification.')

        self.class_dict={classes[0]:-1,classes[1]:1}
        self.rev_class_dict = {-1:classes[0] ,  1:classes[1]}
        y=copy.copy(y)
        for _ in range(X.shape[0]):
            y[_]=self.class_dict[y[_]]

        self.X=np.vstack([X,unlabeled_X])
        Y=np.diag(y)
        if self.distance_function == 'knn':
            if self.neighbor_mode=='connectivity':
                W = kneighbors_graph(self.X, self.n_neighbor, mode='connectivity',include_self=False)
                W = (((W + W.T) > 0) * 1).todense()
            else:
                W = kneighbors_graph(self.X, self.n_neighbor, mode='distance',include_self=False)
                W = W.maximum(W.T).todense()
                W = np.exp(-W**2/4/self.t)
        elif self.distance_function == 'linear':
            W=linear_kernel(self.X,self.X)
        elif self.distance_function =='rbf':
            W=rbf_kernel(self.X,self.X,self.gamma_d)

        elif callable(self.distance_function):
            if self.gamma_d is not None:
                W=self.distance_function(self.X,self.X,self.gamma_d)
            else:
                W = self.distance_function(self.X, self.X)
        else:
            W=rbf_kernel(self.X,self.X,self.gamma_d)
        L = sparse.csr_matrix(np.diag(np.array(W.sum(0))) - W)

        if self.kernel_function == 'rbf':
            K = rbf_kernel(self.X,self.X,self.gamma_k)
        elif self.kernel_function == 'linear':
            K=linear_kernel(self.X,self.X)
        elif callable(self.kernel_function):
            if self.gamma_k is not None:
                K = self.kernel_function(self.X,self.X,self.gamma_k)
            else:
                K = self.kernel_function(self.X, self.X)
        else:
            K = rbf_kernel(self.X, self.X, self.gamma_k)

        num_labeled=X.shape[0]
        num_unlabeled=unlabeled_X.shape[0]
        J = np.concatenate([np.identity(num_labeled), np.zeros(num_labeled * num_unlabeled).reshape(num_labeled, num_unlabeled)], axis=1)
        alpha_star = np.linalg.inv(2 * self.gamma_A * np.identity(num_labeled + num_unlabeled) \
                                     + ((2 * self.gamma_I) / (num_labeled + num_unlabeled) ** 2) * L.dot(K)).dot(J.T).dot(Y)
        Q = Y.dot(J).dot(K).dot(alpha_star)
        Q = (Q+Q.T)/2

        e = np.ones(num_labeled)
        q = -e

        def objective_fun(x):
            return (1 / 2) * x.dot(Q).dot(x) + q.dot(x)

        def objective_jac(x):
            return np.squeeze(np.array(x.T.dot(Q) + q))

        def constraints_fun(x):
            return x.dot(np.diag(Y))

        def constraints_jac(x):
            return np.diag(Y)

        bounds = [(0, 1 / num_labeled) for _ in range(num_labeled)]

        constraints = {'type': 'eq', 'fun': constraints_fun, 'jac': constraints_jac}

        beta_star = minimize(objective_fun, np.zeros(num_labeled), jac=objective_jac, constraints=constraints, bounds=bounds)['x']
        self.alpha = alpha_star.dot(beta_star)

        if self.kernel_function == 'rbf':
            K = rbf_kernel(self.X,X,self.gamma_k)
        elif self.kernel_function == 'linear':
            K= linear_kernel(self.X,X)
        elif callable(self.kernel_function):
            if self.gamma_k is not None:
                K = self.kernel_function(self.X,X,self.gamma_k)
            else:
                K = self.kernel_function(self.X,X)
        else:
            K = rbf_kernel(self.X, X, self.gamma_k)
        f = np.squeeze(np.array(self.alpha)).dot(K)
        idx_super_vectors=np.nonzero((beta_star-1e-8>0)*(1/num_labeled-beta_star>1e-8))[0]
        try:
            idx = idx_super_vectors[0]
        except:
            idx=0
        self.bound=np.diag(Y)[idx]-f[idx]
        return self


    def decision_function(self,X):
        if self.kernel_function == 'rbf':
            K = rbf_kernel(self.X,X,self.gamma_k)
        elif self.kernel_function == 'linear':
            K = linear_kernel(self.X, X)
        elif callable(self.kernel_function):
            if self.gamma_k is not None:
                K = self.kernel_function(self.X,X,self.gamma_k)
            else:
                K = self.kernel_function(self.X,X)
        else:
            K = rbf_kernel(self.X, X, self.gamma_k)
        f = np.squeeze(np.array(self.alpha)).dot(K)
        return f+self.bound

    def predict_proba(self,X):
        y_desision = self.decision_function(X)
        y_score = np.full((X.shape[0], 2), 0, np.float)
        y_score[:,0]=1/(1+np.exp(y_desision))
        y_score[:, 1] =1- y_score[:,0]
        return y_score

    def predict(self,X):
        y_desision = self.decision_function(X)
        y_pred = np.ones(X.shape[0])
        y_pred[y_desision < 0] = -1
        for _ in range(X.shape[0]):
            y_pred[_]=self.rev_class_dict[y_pred[_]]
        return y_pred

    def evaluate(self,X,y=None):

        if isinstance(X,Dataset) and y is None:
            y=getattr(X,'y')

        self.y_score = self.predict_proba(X)
        self.y_pred=self.predict(X)

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