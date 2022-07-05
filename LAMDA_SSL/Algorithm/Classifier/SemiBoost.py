import numpy as np
from sklearn import neighbors
from sklearn.svm import SVC
import copy
from scipy import sparse
from sklearn.metrics.pairwise import rbf_kernel
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
import inspect
from torch.utils.data.dataset import Dataset
import LAMDA_SSL.Config.SemiBoost as config

class SemiBoost(InductiveEstimator,ClassifierMixin):
    # Binary
    def __init__(self, base_estimator = config.base_estimator,
                        n_neighbors=config.n_neighbors, n_jobs = config.n_jobs,
                        max_models = config.max_models,
                        sample_percent = config.sample_percent,
                        sigma_percentile = config.sigma_percentile,
                        similarity_kernel = config.similarity_kernel,gamma=config.gamma,
                        evaluation=config.evaluation,verbose=config.verbose,file=config.file):
        # >> Parameter:
        # >> - base_estimator: The base supervised learner used in the algorithm.
        # >> - similarity_kernel: 'rbf'、'knn' or callable. Specifies the kernel type to be used in the algorithm.
        # >> - n_neighbors: It is valid when the kernel function is 'knn', indicating the value of k in the k nearest neighbors.
        # >> - n_jobs: It is valid when the kernel function is 'knn', indicating the number of parallel jobs.
        # >> - gamma: It is valid when the kernel function is 'rbf', indicating the gamma value of the rbf kernel.
        # >> - max_models: The most number of models in the ensemble.
        # >> - sample_percent: The number of samples sampled at each iteration as a proportion of the remaining unlabeled samples.
        # >> - sigma_percentile: Scale parameter used in the 'rbf' kernel.
        self.BaseModel = base_estimator
        self.n_neighbors=n_neighbors
        self.n_jobs=n_jobs
        self.max_models=max_models
        self.sample_percent=sample_percent
        self.sigma_percentile=sigma_percentile
        self.similarity_kernel=similarity_kernel
        self.gamma=gamma
        self.evaluation = evaluation
        self.verbose = verbose
        self.file = file
        self.y_pred=None
        self.y_score=None
        self._estimator_type = ClassifierMixin._estimator_type

    def fit(self, X, y,unlabeled_X):
        classes, y_indices = np.unique(y, return_inverse=True)

        self.class_dict={classes[0]:-1,classes[1]:1}
        self.rev_class_dict = {-1:classes[0] ,  1:classes[1]}
        y=copy.copy(y)
        for _ in range(X.shape[0]):
            y[_]=self.class_dict[y[_]]

        # Localize labeled data
        num_labeled=X.shape[0]
        num_unlabeled=unlabeled_X.shape[0]

        # The parameter C is defined in the paper as C = num_labeled/num_labeled

        idx=np.arange(num_labeled+num_unlabeled)
        idx_label=idx[:num_labeled]
        idx_not_label=idx[num_labeled:]

        X_all=np.concatenate((X,unlabeled_X))
        y_all=np.concatenate((y,np.zeros(num_unlabeled,dtype=int)))
        # First we need to create the similarity matrix
        if self.similarity_kernel == 'knn':

            self.S = neighbors.kneighbors_graph(X_all,
                                                n_neighbors=self.n_neighbors,
                                                mode='distance',
                                                include_self=True,
                                                n_jobs=self.n_jobs)

            self.S = sparse.csr_matrix(self.S)

        elif self.similarity_kernel == 'rbf':
            self.S = np.sqrt(rbf_kernel(X_all, gamma = self.gamma))

            sigma = np.percentile(np.log(self.S), self.sigma_percentile)
            sigma_2 = (1/sigma**2)*np.ones((self.S.shape[0],self.S.shape[0]))
            self.S = np.power(self.S, sigma_2)
            # Matrix to sparse
            self.S = sparse.csr_matrix(self.S)

        elif self.similarity_kernel is not None:
            if 'gamma' in inspect.getfullargspec(self.similarity_kernel).args:
                self.S = self.similarity_kernel(X_all,X_all,gamma=self.gamma)
            else:
                self.S = self.similarity_kernel(X_all,X_all)
            self.S = sparse.csr_matrix(self.S)
        else:
            raise ValueError('No such kernel!')

        # Initialise variables
        self.models = []
        self.weights = []
        H = np.zeros(num_unlabeled)

        # Loop for adding sequential models
        for t in range(self.max_models):
            # Calculate p_i and q_i for every sample

            p_1 = np.einsum('ij,j', self.S[:,idx_label].todense(), (y_all[idx_label]==1))[idx_not_label]*np.exp(-2*H)
            p_2 = np.einsum('ij,j', self.S[:,idx_not_label].todense(), np.exp(H))[idx_not_label]*np.exp(-H)
            p = np.add(p_1, p_2)
            # print('p')
            # print(p.shape)
            p = np.asarray(p)

            q_1 = np.einsum('ij,j', self.S[:,idx_label].todense(), (y_all[idx_label]==-1))[idx_not_label]*np.exp(2*H)
            q_2 = np.einsum('ij,j', self.S[:,idx_not_label].todense(), np.exp(-H))[idx_not_label]*np.exp(H)
            q = np.add(q_1, q_2)
            # print('q')

            q = np.asarray(q)

            #=============================================================
            # Compute predicted label z_i
            #=============================================================
            z = np.sign(p-q)

            # print(z.shape)
            z_conf = np.abs(p-q)
            #=============================================================
            # Sample sample_percent most confident predictions
            #=============================================================
            # Sampling weights

            # If there are non-zero weights
            sample_weights = z_conf / np.sum(z_conf)
            if np.any(sample_weights != 0):


                idx_aux = np.random.choice(np.arange(len(z)),
                                              size = int(self.sample_percent*len(idx_not_label)),
                                              p = sample_weights,
                                              replace = False)
                idx_sample = idx_not_label[idx_aux]

            else:
                break

            # Create new X_t, y_t
            idx_total_sample = np.concatenate([idx_label,idx_sample])
            X_t = X_all[idx_total_sample,]
            np.put(y_all, idx_sample, z[idx_aux])# Include predicted to train new model
            y_t = y_all[idx_total_sample]

            #=============================================================
            # Fit BaseModel to samples using predicted labels
            #=============================================================
            # Fit model to unlabeled observations
            clf = self.BaseModel
            clf.fit(X_t, y_t)
            # Make predictions for unlabeled observations
            h = clf.predict(X_all[idx_not_label])

            # Refresh indexes
            idx_label = idx_total_sample
            idx_not_label = np.array([i for i in np.arange(len(y_all)) if i not in idx_label])


            #=============================================================
            # Compute weight (a) for the BaseModel as in (12)
            #=============================================================
            e = (np.dot(p,h==-1) + np.dot(q,h==1))/(np.sum(np.add(p,q)))
            a = 0.25*np.log((1-e)/e)
            #=============================================================
            # Update final model
            #=============================================================
            # If a<0 the model is not converging
            if a<0:
                break

            # Save model
            self.models.append(clf)
            # Save weights
            self.weights.append(a)
            # Update
            H = np.zeros(len(idx_not_label))
            for i in range(len(self.models)):
                H = np.add(H, self.weights[i]*self.models[i].predict(X_all[idx_not_label]))

            if len(idx_not_label) == 0:
                break
        self.unlabeled_X=unlabeled_X
        self.unlabeled_y=y_all[num_unlabeled:]

    def predict_proba(self, X):
        y_proba = np.full((X.shape[0], 2), 0, np.float)
        for i in range(len(self.models)):
            y_proba = np.add(y_proba, self.weights[i] * self.models[i].predict_proba(X))
        return y_proba

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        # Predict weighting each model
        # w = np.sum(self.weights)
        for i in range(len(self.models)):
            # estimate = np.add(estimate,  self.weights[i]*self.models[i].predict_proba(X)[:,1]/w)
            y_pred = np.add(y_pred, self.weights[i]*self.models[i].predict(X))
        y_pred = np.array(list(1 if x>0 else -1 for x in y_pred))
        y_pred = y_pred.astype(int)
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