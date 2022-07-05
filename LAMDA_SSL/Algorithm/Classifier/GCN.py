import copy

from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
from LAMDA_SSL.Opitimizer.BaseOptimizer import BaseOptimizer
import LAMDA_SSL.Network.GCN as GCNNET
from torch.utils.data.dataset import Dataset
from torch_geometric.data.data import Data
import torch
from LAMDA_SSL.utils import class_status
import LAMDA_SSL.Config.GCN as config
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy

class GCN(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 num_features=config.num_features,
                 num_classes=config.num_classes,
                 normalize=config.normalize,
                 epoch=config.epoch,
                 eval_epoch=config.eval_epoch,
                 optimizer=config.optimizer,
                 weight_decay=config.weight_decay,
                 scheduler=config.scheduler,
                 device=config.device,
                 evaluation=config.evaluation,
                 network=config.network,
                 parallel=config.parallel,
                 file=config.file,
                 verbose=config.verbose
                 ):
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
        self.normalize=normalize
        self.num_features=num_features
        self.num_classes=num_classes
        self._estimator_type = ClassifierMixin._estimator_type

    def fit(self,X=None,y=None,unlabeled_X=None,valid_X=None,valid_y=None,
            edge_index=None,train_mask=None,labeled_mask=None,unlabeled_mask=None,valid_mask=None,test_mask=None):
        self.init_train_dataset(X,y,unlabeled_X,edge_index,train_mask,labeled_mask,unlabeled_mask,valid_mask,test_mask)
        self.init_train_dataloader()
        self.start_fit()
        self.epoch_loop(valid_X,valid_y)
        self.end_fit()
        return self

    def start_fit(self):
        self.num_features= self.data.x.shape[1] if self.num_features is None else self.num_features
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self.data.y).num_classes
        if self.network is None:
            self.network=GCNNET.GCN(num_features=self.num_features,num_classes=self.num_classes,
                                                                    normalize=self.normalize)
            self._network=copy.deepcopy(self.network)
            self.init_model()
            self.init_ema()
            self.init_optimizer()
            self.init_scheduler()
        self._network.zero_grad()
        self._network.train()

    def init_optimizer(self):
        if isinstance(self._optimizer,BaseOptimizer):
            grouped_parameters=[
                dict(params=self._network.conv1.parameters(), weight_decay=self.weight_decay),
                dict(params=self._network.conv2.parameters(), weight_decay=0)
            ]
            self._optimizer=self._optimizer.init_optimizer(params=grouped_parameters)

    def init_train_dataloader(self):
        pass

    def init_train_dataset(self, X=None, y=None, unlabeled_X=None,
                           edge_index=None,train_mask=None,labeled_mask=None,
                           unlabeled_mask=None,val_mask=None,test_mask=None):
        if isinstance(X,Dataset):
            X=X.data
        if not isinstance(X,Data):
            if unlabeled_X is not None:
                if not isinstance(X, torch.Tensor):
                    X = torch.Tensor(X)
                if not isinstance(y, torch.Tensor):
                    y = torch.LongTensor(y)
                if not isinstance(unlabeled_X, torch.Tensor):
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

            X=Data(X=X,y=y,edge_index=edge_index,train_mask=train_mask,labeled_mask=labeled_mask,
                   unlabeled_mask=unlabeled_mask,val_mask=val_mask,test_mask=test_mask)
        self.data=X
        self.labeled_mask = self.data.labeled_mask if hasattr(self.data,'labeled_mask') else None
        self.unlabeled_mask = self.data.unlabeled_mask if hasattr(self.data,'unlabeled_mask') else None
        self.valid_mask = self.data.val_mask if hasattr(self.data, 'val_mask') else None
        self.test_mask = self.data.test_mask if hasattr(self.data, 'test_mask') else None



    def epoch_loop(self, valid_X=None, valid_y=None):
        self.data=self.data.to(self.device)
        if valid_X is None:
            valid_X=self.data.val_mask

        for self._epoch in range(1,self.epoch+1):
            if self.verbose:
                print(self._epoch,file=self.file)
            train_result = self.train(lb_X=self.data.labeled_mask)

            self.end_batch_train(train_result)

            if valid_X is not None and self.eval_epoch is not None and self._epoch % self.eval_epoch==0:
                self.evaluate(X=valid_X,y=valid_y,valid=True)



    def train(self, lb_X=None, lb_y=None, ulb_X=None, lb_idx=None, ulb_idx=None, *args, **kwargs):
        self.logits = self._network(self.data)
        lb_logits = self.logits[lb_X]
        lb_y=self.data.y[lb_X]
        return lb_logits,lb_y

    def get_loss(self,train_result,*args,**kwargs):
        lb_logits, lb_y=train_result
        loss=Cross_Entropy(reduction='mean')(lb_logits,lb_y)
        return loss


    def init_pred_dataloader(self,valid=False):
        pass

    def init_pred_dataset(self, X=None, valid=False):
        if X is not None and not isinstance(X, torch.Tensor):
            X = torch.BoolTensor(X).to(self.device)
        if valid:
            self.pred_mask = X if X is not None else self.data.val_mask
        else:
            self.pred_mask = X if X is not None else self.data.test_mask

    def pred_batch_loop(self):
        with torch.no_grad():
            self.y_est=self.logits[self.pred_mask]


    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)


    @torch.no_grad()
    def evaluate(self,X=None,y=None,valid=False):
        y_pred=self.predict(X,valid=valid)
        y_score=self.y_score
        y =self.data.y[X].cpu().detach().numpy() if y is None else y
        if self.evaluation is None:
            return None
        elif isinstance(self.evaluation,(list,tuple)):
            result=[]
            for eval in self.evaluation:
                score=eval.scoring(y,y_pred,y_score)
                if self.verbose:
                    print(score,file=self.file)
                result.append(score)
            self.result=result
            return result
        elif isinstance(self.evaluation,dict):
            result={}
            for key,val in self.evaluation.items():
                result[key]=val.scoring(y,y_pred,y_score)
                if self.verbose:
                    print(key,' ',result[key],file=self.file)
            self.result = result
            return result
        else:
            result=self.evaluation.scoring(y,y_pred,y_score)
            if self.verbose:
                print(result,file=self.file)
            self.result = result
            return result