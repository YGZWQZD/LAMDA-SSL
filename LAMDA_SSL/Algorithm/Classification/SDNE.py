import copy
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from torch.utils.data.dataset import Dataset
from torch_geometric.data.data import Data
from sklearn.base import ClassifierMixin
import LAMDA_SSL.Network.SDNE as SDNENET
import scipy.sparse as sparse
import torch
import LAMDA_SSL.Config.SDNE as config

class SDNE(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 base_estimator=config.base_estimator,
                 alpha=config.alpha,
                 beta=config.beta,
                 gamma=config.gamma,
                 xeqs=config.xeqs,
                 dim_in=config.dim_in,
                 num_nodes=config.num_nodes,
                 hidden_layers=config.hidden_layers,
                 weight_decay=config.weight_decay,
                 epoch=config.epoch,
                 eval_epoch=config.eval_epoch,
                 device=config.device,
                 network=config.network,
                 optimizer=config.optimizer,
                 scheduler=config.scheduler,
                 parallel=config.parallel,
                 evaluation=config.evaluation,
                 file=config.file,
                 verbose=config.verbose
                 ):
        # >> Parameter:
        # >> - xeqs: Whether to use the adjacency matrix as the feature matrix of the node.
        # >> - dim_in: The dimension of node features. It is valid when xeqs is False.
        # >> - num_nodes: The number of nodes.
        # >> - hidden_layers: Encoder hidden layer dimension.
        # >> - alpha: The weight of Laplacian regularization.
        # >> - gamma: The weight of L2 regularation.
        # >> - beta: The weight of the edges in the graph that are not 0 in the loss of consistency between the input and output of the autoencoder.
        # >> - base_estimator: A supervised learner that classifies using the node features obtained by the encoder.
        DeepModelMixin.__init__(self,
                                    epoch=epoch,
                                    weight_decay=weight_decay,
                                    network=network,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    device=device,
                                    eval_epoch=eval_epoch,
                                    evaluation=evaluation,
                                    parallel=parallel,
                                    file=file,
                                    verbose=verbose
                                    )
        self.num_nodes=num_nodes
        self.hidden_layers=hidden_layers
        self.dim_in=dim_in
        self.alpha=alpha
        self.beta=beta
        self.xeqs=xeqs
        self.gamma=gamma
        self.base_estimator=base_estimator
        self._estimator_type = ClassifierMixin._estimator_type

    def fit(self,X=None,y=None,unlabeled_X=None,valid_X=None,valid_y=None,
            edge_index=None,train_mask=None,labeled_mask=None,unlabeled_mask=None,valid_mask=None,test_mask=None):
        self.init_train_dataset(X,y,unlabeled_X,edge_index,train_mask,labeled_mask,unlabeled_mask,valid_mask,test_mask)
        self.init_train_dataloader()
        self.start_fit()
        self.fit_epoch_loop(valid_X,valid_y)
        self.end_fit()
        return self

    def start_fit(self):
        self.num_features= self.data.x.shape[1] if self.dim_in is None else self.dim_in
        self.num_nodes=self.data.x.shape[0] if self.num_nodes is None else self.num_nodes
        if self.network is None:
            self.network=SDNENET.SDNE(dim_in=self.num_nodes,hidden_layers=self.hidden_layers) if self.xeqs else \
                SDNENET.SDNE(dim_in=self.num_features,hidden_layers=self.hidden_layers)
            self._network=copy.deepcopy(self.network)
            self.init_model()
            self.init_ema()
            self.init_optimizer()
            self.init_scheduler()
        self._network.zero_grad()
        self._network.train()

    def init_train_dataloader(self):
        pass

    def init_train_dataset(self, X=None, y=None, unlabeled_X=None,
                           edge_index=None,train_mask=None,labeled_mask=None,
                           unlabeled_mask=None,val_mask=None,test_mask=None):
        self._train_dataset = copy.deepcopy(self.train_dataset)
        if isinstance(X,Dataset):
            X=X.data
        if not isinstance(X,Data):
            if not isinstance(X, torch.Tensor):
                X = torch.Tensor(X)
            if not isinstance(y, torch.Tensor):
                y = torch.LongTensor(y)
            if unlabeled_X is not None and not isinstance(unlabeled_X, torch.Tensor):
                unlabeled_X = torch.Tensor(unlabeled_X)
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.LongTensor(edge_index)
            if not isinstance(train_mask, torch.Tensor):
                train_mask = torch.BoolTensor(train_mask)
            if not isinstance(labeled_mask, torch.Tensor):
                labeled_mask = torch.BoolTensor(labeled_mask)
            if not isinstance(unlabeled_mask, torch.Tensor):
                unlabeled_mask = torch.BoolTensor(unlabeled_mask)
            if not isinstance(val_mask, torch.Tensor):
                val_mask = torch.BoolTensor(val_mask)
            if not isinstance(val_mask, torch.Tensor):
                test_mask = torch.BoolTensor(test_mask)

            if unlabeled_X is not None:
                X = torch.cat((X, unlabeled_X), dim=0)
                unlabeled_y = torch.ones(unlabeled_X.shape[0]) * -1
                y = torch.cat((y, unlabeled_y), dim=0)

            X=Data(x=X,y=y,edge_index=edge_index,train_mask=train_mask,labeled_mask=labeled_mask,
                   unlabeled_mask=unlabeled_mask,val_mask=val_mask,test_mask=test_mask)
        self.data=X.to(self.device)
        self.train_mask = self.data.train_mask.to(self.device) if hasattr(self.data, 'train_mask') else None
        self.labeled_mask = self.data.labeled_mask.to(self.device) if hasattr(self.data,'labeled_mask') else None
        self.unlabeled_mask = self.data.unlabeled_mask.to(self.device) if hasattr(self.data,'unlabeled_mask') else None
        self.valid_mask = self.data.val_mask.to(self.device) if hasattr(self.data, 'val_mask') else None
        self.test_mask = self.data.test_mask.to(self.device) if hasattr(self.data, 'test_mask') else None
        adjacency_matrix, laplace_matrix = self.create_adjacency_laplace_matrix()
        self.adjacency_matrix = torch.from_numpy(adjacency_matrix.toarray()).float().to(self.device)
        self.laplace_matrix = torch.from_numpy(laplace_matrix.toarray()).float().to(self.device)

    def estimator_fit(self):
        X=self.embedding[self.data.labeled_mask] if hasattr(self.data,'labeled_mask') and  self.data.labeled_mask is not None \
            else self.embedding[self.data.train_mask]
        y=self.data.y[self.labeled_mask] if hasattr(self.data,'labeled_mask') and  self.data.labeled_mask is not None \
            else self.data.y[self.data.train_mask]

        X=X.cpu().detach().numpy()
        y=y.cpu().detach().numpy()
        self.base_estimator.fit(X,y)

    def create_adjacency_laplace_matrix(self):
        self.edge_index=self.data.edge_index
        adjacency_matrix_data = []
        adjacency_matrix_row_index = []
        adjacency_matrix_col_index = []
        self.num_node=self.data.x.shape[0]
        for _ in range(self.edge_index.shape[1]):
            adjacency_matrix_data.append(1.0)
            adjacency_matrix_row_index.append(self.edge_index[0][_])
            adjacency_matrix_col_index.append(self.edge_index[1][_])

        adjacency_matrix = sparse.csr_matrix((adjacency_matrix_data,
                                              (adjacency_matrix_row_index, adjacency_matrix_col_index)),
                                             shape=(self.num_node, self.num_node))
        # L = D - A
        # Calculate degrees
        adjacency_matrix_ = sparse.csr_matrix((adjacency_matrix_data+adjacency_matrix_data,
                                               (adjacency_matrix_row_index+adjacency_matrix_col_index,
                                                adjacency_matrix_col_index+adjacency_matrix_row_index)),
                                              shape=(self.num_node, self.num_node))
        degree_matrix = sparse.diags(adjacency_matrix_.sum(axis=1).flatten().tolist()[0])
        laplace_matrix = degree_matrix - adjacency_matrix_
        return adjacency_matrix, laplace_matrix

    def end_fit_epoch(self, train_result,*args, **kwargs):
        self.loss = self.get_loss(train_result)
        self.optimize(self.loss)

    def fit_epoch_loop(self, valid_X=None, valid_y=None):
        self.valid_performance = {}
        self.data=self.data.to(self.device)
        if valid_X is None:
            valid_X=self.data.val_mask
        for self._epoch in range(1,self.epoch+1):
            print(self._epoch,file=self.file)
            train_performance = self.train(lb_X=self.data.labeled_mask)
            self.end_fit_epoch(train_performance)
            if valid_X is not None and self.eval_epoch is not None and self._epoch % self.eval_epoch==0:
                self.estimator_fit()
                self.evaluate(X=valid_X,y=valid_y,valid=True)
                self.valid_performance.update({"epoch_" + str(self._epoch): self.performance})

        if valid_X is not None and (self.eval_epoch is None or self.epoch% self.eval_epoch!=0):
            self.estimator_fit()
            self.evaluate(X=valid_X, y=valid_y, valid=True)
            self.valid_performance.update({"epoch_" + str(self._epoch): self.performance})

    def end_fit(self):
        self.estimator_fit()

    def train(self, lb_X=None, lb_y=None, ulb_X=None, lb_idx=None, ulb_idx=None, *args, **kwargs):
        if self.xeqs:
            X=self.adjacency_matrix
        else:
            X=self.data.x
        Y,X_hat = self._network(X=X)
        self.embedding=Y
        return X,X_hat,Y

    def get_loss(self,train_result,*args,**kwargs):
        X,X_hat,Y=train_result

        if self.xeqs:
            beta_matrix = torch.ones_like(self.adjacency_matrix)
            mask = self.adjacency_matrix != 0
            beta_matrix[mask] = self.beta
            loss_2nd = torch.mean(torch.sum(torch.pow((X - X_hat) * beta_matrix, 2), dim=1))
        else:
            loss_2nd = torch.mean(torch.sum(torch.pow((X - X_hat) , 2), dim=1))
        L_reg = 0
        for param in self._network.parameters():
            L_reg += self.gamma * torch.sum(param * param)
        loss_1st =  self.alpha * 2 * torch.trace(torch.matmul(torch.matmul(Y.transpose(0,1), self.laplace_matrix), Y))
        return loss_1st+loss_2nd+L_reg



    def predict(self,X=None,valid=False):
        if X is not None and not isinstance(X, torch.Tensor):
            X = torch.BoolTensor(X).to(self.device)
        if valid:
            X = self.embedding[X] if X is not None else self.embedding[self.data.val_mask]
        else:
            X = self.embedding[X] if X is not None else self.embedding[self.data.test_mask]
        if isinstance(X,torch.Tensor):
            X=X.cpu().detach().numpy()
        return self.base_estimator.predict(X)

    def predict_proba(self, X=None, valid=False):
        if X is not None and not isinstance(X, torch.Tensor):
            X = torch.BoolTensor(X).to(self.device)
        if valid:
            X = self.embedding[X] if X is not None else self.embedding[self.data.val_mask]
        else:
            X = self.embedding[X] if X is not None else self.embedding[self.data.test_mask]
        if isinstance(X,torch.Tensor):
            X=X.cpu().detach().numpy()
        return self.base_estimator.predict_proba(X)

    @torch.no_grad()
    def evaluate(self, X, y=None,valid=False):
        y_pred = self.predict(X,valid=valid)
        if hasattr(self.base_estimator,'predict_proba'):
            y_score = self.predict_proba(X,valid=valid)
        else:
            y_score=None
        self.y_score=y_score
        self.y_pred=y_pred
        if y is not None:
            y=y
        elif valid:
            y = self.data.y[self.data.val_mask].cpu().detach().numpy()
        else:
            y = self.data.y[X].cpu().detach().numpy() if X is not None else self.data.y[self.data.test_mask].cpu().detach().numpy()
        if self.evaluation is None:
            return None
        elif isinstance(self.evaluation, (list, tuple)):
            performance = []
            for eval in self.evaluation:
                score=eval.scoring(y, self.y_pred, self.y_score)
                if self.verbose:
                    print(score, file=self.file)
                performance.append(score)
            self.performance = performance
            return performance
        elif isinstance(self.evaluation, dict):
            performance = {}
            for key, val in self.evaluation.items():
                performance[key] = val.scoring(y, self.y_pred, self.y_score)
                if self.verbose:
                    print(key, ' ', performance[key],file=self.file)
            self.performance = performance
            return performance
        else:
            performance = self.evaluation.scoring(y, self.y_pred, self.y_score)
            if self.verbose:
                print(performance, file=self.file)
            self.performance=performance
            return performance