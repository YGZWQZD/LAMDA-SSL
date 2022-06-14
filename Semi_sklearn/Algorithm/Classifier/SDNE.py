from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from sklearn.base import ClassifierMixin
from Semi_sklearn.Opitimizer.SemiOptimizer import SemiOptimizer
import Semi_sklearn.Network.SDNE as SDNENET
import torch.nn.functional as F
import scipy.sparse as sparse
import torch
from Semi_sklearn.utils import to_device
from sklearn.linear_model import LogisticRegression
class SDNE(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
    def __init__(self,
                 epoch=1,
                 input_dim=None,
                 num_nodes=None,
                 hidden_layers=[250,250],
                 alpha=1e-2,
                 beta=5,
                 gamma=0.9,
                 base_estimator=None,
                 xeqs=True,
                 eval_epoch=None,
                 optimizer=None,
                 scheduler=None,
                 device='cpu',
                 evaluation=None,
                 weight_decay=None,
                 network=None,
                 parallel=None,
                 file=None
                 ):
        if network is not None:
            self.network=network
        else:
            self.network=SDNENET.SDNE(input_dim=num_nodes,hidden_layers=hidden_layers) if xeqs else SDNENET.SDNE(input_dim=input_dim,hidden_layers=hidden_layers)
        SemiDeepModelMixin.__init__(self,
                                    epoch=epoch,
                                    weight_decay=weight_decay,
                                    network=self.network,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    device=device,
                                    eval_epoch=eval_epoch,
                                    evaluation=evaluation,
                                    parallel=parallel,
                                    file=file
                                    )
        self.alpha=alpha
        self.beta=beta
        self.xeqs=xeqs
        self.gamma=gamma
        self.base_estimator=base_estimator if base_estimator is not None else LogisticRegression()
        self._estimator_type = ClassifierMixin._estimator_type

    def init_train_dataloader(self):
        pass

    def init_train_dataset(self, X=None, y=None, unlabeled_X=None):
        self.data=X
        self.labeled_mask = self.data.labeled_mask if hasattr(self.data,'labeled_mask') else None
        self.unlabeled_mask = self.data.labeled_mask if hasattr(self.data,'unlabeled_mask') else None
        adjacency_matrix, laplace_matrix = self.create_adjacency_laplace_matrix()
        self.adjacency_matrix = torch.from_numpy(adjacency_matrix.toarray()).float().to(self.device)
        self.laplace_matrix = torch.from_numpy(laplace_matrix.toarray()).float().to(self.device)



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
        # L = D - A  有向图的度等于出度和入度之和; 无向图的领接矩阵是对称的，没有出入度之分直接为每行之和
        # 计算度数
        adjacency_matrix_ = sparse.csr_matrix((adjacency_matrix_data+adjacency_matrix_data,
                                               (adjacency_matrix_row_index+adjacency_matrix_col_index,
                                                adjacency_matrix_col_index+adjacency_matrix_row_index)),
                                              shape=(self.num_node, self.num_node))
        degree_matrix = sparse.diags(adjacency_matrix_.sum(axis=1).flatten().tolist()[0])
        laplace_matrix = degree_matrix - adjacency_matrix_
        return adjacency_matrix, laplace_matrix

    def epoch_loop(self, valid_X=None, valid_y=None):
        self.data=self.data.to(self.device)
        for self._epoch in range(1,self.epoch+1):
            print(self._epoch,file=self.file)
            train_result = self.train(lb_X=self.data.labeled_mask)
            self.end_batch_train(train_result)

    def end_fit(self):
        X=self.embedding[self.data.labeled_mask] if hasattr(self.data,'labeled_mask') and  self.data.labeled_mask is not None \
            else self.embedding[self.data.train_mask]
        y=self.data.y[self.labeled_mask] if hasattr(self.data,'labeled_mask') and  self.data.labeled_mask is not None \
            else self.data.y[self.data.train_mask]
        # X=self.data.x[self.data.labeled_mask] if hasattr(self.data,'labeled_mask') and  self.data.labeled_mask is not None \
        #     else self.data.x[self.data.train_mask]
        # y=self.data.y[self.labeled_mask] if hasattr(self.data,'labeled_mask') and  self.data.labeled_mask is not None \
        #     else self.data.y[self.data.train_mask]

        X=X.cpu().detach().numpy()
        y=y.cpu().detach().numpy()
        self.base_estimator.fit(X,y)

    def train(self, lb_X=None, lb_y=None, ulb_X=None, lb_idx=None, ulb_idx=None, *args, **kwargs):
        if self.xeqs:
            X=self.adjacency_matrix
        else:
            X=self.data.x
        Y,X_hat = self._network(X=X)
        # print(Y.shape)
        # print(X_hat.shape)
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
        # loss_1st 一阶相似度损失函数 论文公式(9) alpha * 2 *tr(Y^T L Y)
        loss_1st =  self.alpha * 2 * torch.trace(torch.matmul(torch.matmul(Y.transpose(0,1), self.laplace_matrix), Y))
        return loss_1st+loss_2nd+L_reg



    def predict(self,X=None,valid=None):
        X=self.embedding[X] if X is not None else  self.embedding[self.data.unlabeled_mask]
        if isinstance(X,torch.Tensor):
            X=X.cpu().detach().numpy()
        return self.base_estimator.predict(X)

    def predict_proba(self,X=None,valid=None):
        X=self.embedding[X] if X is not None else  self.embedding[self.data.unlabeled_mask]
        if isinstance(X,torch.Tensor):
            X=X.cpu().detach().numpy()
        return self.base_estimator.predict(X)

    @torch.no_grad()
    def evaluate(self, X, y=None, valid=None):
        y_pred = self.predict(X).cpu()
        if hasattr(self.base_estimator,'predict_proba'):
            y_score = self.predict_proba(X).cpu()
        else:
            y_score=None
        y = self.data.y[X] if y is None else y
        if self.evaluation is None:
            return None
        elif isinstance(self.evaluation, (list, tuple)):
            result = []
            for eval in self.evaluation:
                result.append(eval.scoring(y, y_pred, y_score))
            return result
        elif isinstance(self.evaluation, dict):
            result = {}
            for key, val in self.evaluation.items():
                # print(y.shape)
                # print(y_pred.shape)
                result[key] = val.scoring(y, y_pred, y_score)
                print(key, ' ', result[key])
            return result
        else:
            result = self.evaluation.scoring(y, y_pred, y_score)
            return result