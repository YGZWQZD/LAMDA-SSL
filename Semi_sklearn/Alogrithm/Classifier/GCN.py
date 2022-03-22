from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from sklearn.base import ClassifierMixin
from Semi_sklearn.Opitimizer.SemiOptimizer import SemiOptimizer
import Semi_sklearn.Network.GCN as GCNNET
import torch.nn.functional as F
import torch
from Semi_sklearn.utils import to_device
class GCN(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
    def __init__(self,
                 epoch=1,
                 num_features=1433,
                 num_classes=7,
                 normalize=True,
                 eval_epoch=None,
                 optimizer=None,
                 scheduler=None,
                 device='cpu',
                 evaluation=None,
                 weight_decay=None,
                 network=None
                 ):
        self.network=network if network is not None else GCNNET.GCN(num_features=num_features,num_classes=num_classes,
                                                                    normalize=normalize)
        SemiDeepModelMixin.__init__(self,
                                    epoch=epoch,
                                    weight_decay=weight_decay,
                                    network=self.network,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    device=device,
                                    eval_epoch=eval_epoch,
                                    evaluation=evaluation
                                    )
        self._estimator_type = ClassifierMixin._estimator_type

    def init_optimizer(self):
        if isinstance(self._optimizer,SemiOptimizer):
            grouped_parameters=[
                dict(params=self._network.conv1.parameters(), weight_decay=5e-4),
                dict(params=self._network.conv2.parameters(), weight_decay=0)
            ]
            self._optimizer=self._optimizer.init_optimizer(params=grouped_parameters)

    def init_train_dataloader(self):
        pass

    def init_train_dataset(self, X=None, y=None, unlabeled_X=None):
        self.data=X
        self.labeled_mask = self.data.labeled_mask if hasattr(self.data,'labeled_mask') else None
        self.unlabeled_mask = self.data.labeled_mask if hasattr(self.data,'unlabeled_mask') else None



    def epoch_loop(self, valid_X=None, valid_y=None):
        self.data=self.data.to(self.device)
        if valid_X is None:
            valid_X=self.data.val_mask

        for self._epoch in range(1,self.epoch+1):
            print(self._epoch)
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
        loss=F.nll_loss(F.log_softmax(lb_logits,dim=1), lb_y)
        return loss


    def init_pred_dataloader(self,valid=False):
        pass

    def init_pred_dataset(self, X=None, valid=False):
        if X is not None:
            X=to_device(X,self.device)
        if valid:
            self.pred_mask = X if X is not None else self.data.val_mask
        else:
            self.pred_mask = X if X is not None else self.data.test_mask

    def pred_batch_loop(self):
        with torch.no_grad():
            self.y_est=self.logits[self.pred_mask]


    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)


    @torch.no_grad()
    def evaluate(self,X,y=None,valid=False):
        y_pred=self.predict(X,valid=valid).cpu()
        y_score=self.y_score.cpu()
        y =self.data.y[X] if y is None else y
        if self.evaluation is None:
            return None
        elif isinstance(self.evaluation,(list,tuple)):
            result=[]
            for eval in self.evaluation:
                result.append(eval.scoring(y,y_pred,y_score))
            return result
        elif isinstance(self.evaluation,dict):
            result={}
            for key,val in self.evaluation.items():
                # print(y.shape)
                # print(y_pred.shape)
                result[key]=val.scoring(y,y_pred,y_score)
                print(key,' ',result[key])
            return result
        else:
            result=self.evaluation.scoring(y,y_pred,y_score)
            return result



