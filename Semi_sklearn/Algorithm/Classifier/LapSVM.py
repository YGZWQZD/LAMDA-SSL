import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel
import copy
import inspect

class LapSVM(InductiveEstimator,ClassifierMixin):
    def __init__(self,
           distance_function= rbf_kernel,
           gamma_d=0.01,
           neighbor_mode =None,
           n_neighbor= 5,
           kernel_function= rbf_kernel,
           gamma_k=0.01,
           gamma_A= 0.03125,
           gamma_I= 0):
        self.distance_function=distance_function
        self.neighbor_mode=neighbor_mode
        self.n_neighbor=n_neighbor
        # self.t=t
        self.kernel_function=kernel_function
        self.gamma_k=gamma_k
        self.gamma_d=gamma_d
        self.gamma_A=gamma_A
        self.gamma_I=gamma_I
        self._estimator_type = ClassifierMixin._estimator_type


    def fit(self,X,y,unlabeled_X):
        classes, y_indices = np.unique(y, return_inverse=True)
        if len(classes)!=2:
            raise ValueError('TSVM can only be used in binary classification.')
        # print(classes)

        self.class_dict={classes[0]:-1,classes[1]:1}
        self.rev_class_dict = {-1:classes[0] ,  1:classes[1]}
        y=copy.copy(y)
        for _ in range(X.shape[0]):
            y[_]=self.class_dict[y[_]]


        #construct graph

        self.X=np.vstack([X,unlabeled_X])
        Y=np.diag(y)
        if self.neighbor_mode=='connectivity':
            W = kneighbors_graph(self.X, self.n_neighbor, mode='connectivity',include_self=False)
            W = (((W + W.T) > 0) * 1)
            # print(W.shape)
            # print(type(W))
            # print(W.sum(0).shape)
            # L = sparse.diags(np.array(W.sum(0))[0]).tocsr() - W
        elif self.neighbor_mode=='distance':
            W = kneighbors_graph(self.X, self.n_neighbor, mode='distance',include_self=False)
            W = W.maximum(W.T)
            W = sparse.csr_matrix((np.exp(-W.data**2/4/self.t),W.indices,W.indptr),shape=(self.X.shape[0],self.X.shape[0]))
            # print(W.shape)
            # print(type(W))
            # print(W.sum(0).shape)
            # L = sparse.diags(np.array(W.sum(0))[0]).tocsr() - W
        elif self.distance_function is not None:
            if 'gamma' in inspect.getfullargspec(self.distance_function).args:
                W=self.distance_function(self.X,self.X,self.gamma_d)
            else:
                W = self.distance_function(self.X, self.X)
            W=sparse.csr_matrix(W)
            # print(W.shape)
            # print(type(W))
            # print(W.sum(0).shape)


        else:
            raise Exception()

        # Computing Graph Laplacian
        # print(W.sum(0).shape)
        L = sparse.diags(np.array(W.sum(0))[0]).tocsr() - W
        # L=np.array(W.sum(0))[0]
        # print(L)
        # L=sparse.diags(L)
        # L=L.tocsr()
        # L=L-W
        # Computing K with k(i,j) = kernel(i, j)
        if 'gamma' in inspect.getfullargspec(self.kernel_function).args:
            K = self.kernel_function(self.X,self.X,self.gamma_k)
        else:
            K = self.kernel_function(self.X, self.X)
        l=X.shape[0]
        u=unlabeled_X.shape[0]
        # Creating matrix J [I (l x l), 0 (l x (l+u))]
        J = np.concatenate([np.identity(l), np.zeros(l * u).reshape(l, u)], axis=1)

        # Computing "almost" alpha
        almost_alpha = np.linalg.inv(2 * self.gamma_A * np.identity(l + u) \
                                     + ((2 * self.gamma_I) / (l + u) ** 2) * L.dot(K)).dot(J.T).dot(Y)

        # Computing Q
        Q = Y.dot(J).dot(K).dot(almost_alpha)
        Q = (Q+Q.T)/2

        del W, L, K, J

        e = np.ones(l)
        q = -e

        # ===== Objectives =====
        def objective_func(beta):
            return (1 / 2) * beta.dot(Q).dot(beta) + q.dot(beta)

        def objective_grad(beta):
            return np.squeeze(np.array(beta.T.dot(Q) + q))

        # =====Constraint(1)=====
        #   0 <= beta_i <= 1 / l
        bounds = [(0, 1 / l) for _ in range(l)]

        # =====Constraint(2)=====
        #  Y.dot(beta) = 0
        def constraint_func(beta):
            return beta.dot(np.diag(Y))

        def constraint_grad(beta):
            return np.diag(Y)

        cons = {'type': 'eq', 'fun': constraint_func, 'jac': constraint_grad}

        # ===== Solving =====
        x0 = np.zeros(l)

        beta_hat = minimize(objective_func, x0, jac=objective_grad, constraints=cons, bounds=bounds)['x']

        # Computing final alpha
        self.alpha = almost_alpha.dot(beta_hat)

        del almost_alpha, Q

        # Finding optimal decision boundary b using labeled data
        new_K = self.kernel_function(self.X,X,self.gamma_k)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)

        self.sv_ind=np.nonzero((beta_hat>1e-7)*(beta_hat<(1/l-1e-7)))[0]

        ind=self.sv_ind[0]
        self.b=np.diag(Y)[ind]-f[ind]
        return self


    def decision_function(self,X):
        new_K = self.kernel_function(self.X, X, self.gamma_k)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        return f+self.b

    def predict(self,X):
        Y_ = self.decision_function(X)
        Y_pre = np.ones(X.shape[0])
        Y_pre[Y_ < 0] = -1
        for _ in range(X.shape[0]):
            Y_pre[_]=self.rev_class_dict[Y_pre[_]]
        return Y_pre