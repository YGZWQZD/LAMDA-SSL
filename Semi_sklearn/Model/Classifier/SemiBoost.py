import numpy as np
from sklearn import neighbors
from sklearn.svm import SVC
from scipy import sparse
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics.pairwise import rbf_kernel
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin

class SemiBoostClassifier(InductiveEstimator,ClassifierMixin):

    def __init__(self, base_model =SVC(),
                        n_neighbors=4, n_jobs = 1,
                        max_models = 15,
                        sample_percent = 0.5,
                        sigma_percentile = 90,
                        similarity_kernel = 'rbf'):

        self.BaseModel = base_model
        self.n_neighbors=n_neighbors
        self.n_jobs=n_jobs
        self.max_models=max_models
        self.sample_percent=sample_percent
        self.sigma_percentile=sigma_percentile
        self.similarity_kernel=similarity_kernel
        self._estimator_type = ClassifierMixin._estimator_type

    def fit(self, X, y,unlabeled_X):

        ''' Fit model'''
        # Localize labeled data
        num_labeled=X.shape[0]
        num_unlabeled=unlabeled_X.shape[0]

        # The parameter C is defined in the paper as
        # C = num_labeled/num_labeled

        idx=np.range(num_labeled+num_unlabeled)
        idx_label=idx[:num_labeled]
        idx_not_label=idx[num_unlabeled:]

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
            # First aprox
            self.S = np.sqrt(rbf_kernel(X_all, gamma = 1))
            # set gamma parameter as the 15th percentile
            sigma = np.percentile(np.log(self.S), self.sigma_percentile)
            sigma_2 = (1/sigma**2)*np.ones((self.S.shape[0],self.S.shape[0]))
            self.S = np.power(self.S, sigma_2)
            # Matrix to sparse
            self.S = sparse.csr_matrix(self.S)

        else:
            raise ValueError('No such kernel!')


        #=============================================================
        # Initialise variables
        #=============================================================
        self.models = []
        self.weights = []
        H = np.zeros(num_unlabeled)

        # Loop for adding sequential models
        for t in range(self.max_models):
            #=============================================================
            # Calculate p_i and q_i for every sample
            #=============================================================
            p_1 = np.einsum('ij,j', self.S[:,idx_label].todense(), (y[idx_label]==1))[idx_not_label]*np.exp(-2*H)
            p_2 = np.einsum('ij,j', self.S[:,idx_not_label].todense(), np.exp(H))[idx_not_label]*np.exp(-H)
            p = np.add(p_1, p_2)
            p = np.squeeze(np.asarray(p))

            q_1 = np.einsum('ij,j', self.S[:,idx_label].todense(), (y[idx_label]==-1))[idx_not_label]*np.exp(2*H)
            q_2 = np.einsum('ij,j', self.S[:,idx_not_label].todense(), np.exp(-H))[idx_not_label]*np.exp(H)
            q = np.add(q_1, q_2)
            q = np.squeeze(np.asarray(q))

            #=============================================================
            # Compute predicted label z_i
            #=============================================================
            z = np.sign(p-q)
            z_conf = np.abs(p-q)
            #=============================================================
            # Sample sample_percent most confident predictions
            #=============================================================
            # Sampling weights

            # If there are non-zero weights
            if np.any(z_conf != 0):
                sample_weights = z_conf / np.sum(z_conf)
                idx_aux = np.random.choice(np.arange(len(z)),
                                              size = int(self.sample_percent*len(idx_not_label)),
                                              p = sample_weights,
                                              replace = False)
                idx_sample = idx_not_label[idx_aux]

            else:
                # print('No similar unlabeled observations left.')
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
            #save weights
            self.weights.append(a)
            # Update
            H = np.zeros(len(idx_not_label))
            for i in range(len(self.models)):
                H = np.add(H, self.weights[i]*self.models[i].predict(X_all[idx_not_label]))

            #=============================================================
            # Breaking conditions
            #=============================================================

            # Maximum number of models reached
            # if (t==max_models) & verbose:
            #     print('Maximum number of models reached')

            # If no samples are left without label, break
            if len(idx_not_label) == 0:
                # if verbose:
                #     print('All observations have been labeled')
                #     print('Number of iterations: ',t + 1)
                break
        self.unlabeled_X=unlabeled_X
        self.unlabeled_y=y_all[num_unlabeled:]

        # if verbose:
        #     print('\n The model weights are \n')
        #     print(self.weights)



    def predict(self, X):
        estimate = np.zeros(X.shape[0])
        # Predict weighting each model
        w = np.sum(self.weights)
        for i in range(len(self.models)):
            # estimate = np.add(estimate,  self.weights[i]*self.models[i].predict_proba(X)[:,1]/w)
            estimate = np.add(estimate, self.weights[i]*self.models[i].predict(X))
        estimate = np.array(list(map(lambda x: 1 if x>0 else -1, estimate)))
        estimate = estimate.astype(int)
        return estimate