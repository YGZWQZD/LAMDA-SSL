import copy
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from Semi_sklearn.Opitimizer.SemiOptimizer import SemiOptimizer
from Semi_sklearn.Scheduler.SemiScheduler import SemiLambdaLR
from sklearn.base import ClassifierMixin
from torch.nn import Softmax
import torch
from Semi_sklearn.utils import partial
from Semi_sklearn.utils import class_status
from Semi_sklearn.utils import cross_entropy,consistency_loss
import numpy as np

def fix_bn(m,train=False):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if train:
            m.train()
        else:
            m.eval()

class TemporalEnsembling(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
    def __init__(self,train_dataset=None,test_dataset=None,
                 train_dataloader=None,
                 test_dataloader=None,
                 augmentation=None,
                 network=None,
                 train_sampler=None,
                 train_batch_sampler=None,
                 test_sampler=None,
                 test_batch_sampler=None,
                 epoch=1,
                 num_it_epoch=None,
                 num_it_total=None,
                 warmup=None,
                 eval_epoch=None,
                 eval_it=None,
                 optimizer=None,
                 scheduler=None,
                 device='cpu',
                 evaluation=None,
                 lambda_u=None,
                 mu=None,
                 ema_decay=None,
                 weight_decay=None,
                 num_classes=None,
                 num_samples=None
                 ):
        SemiDeepModelMixin.__init__(self,train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    train_dataloader=train_dataloader,
                                    test_dataloader=test_dataloader,
                                    augmentation=augmentation,
                                    network=network,
                                    train_sampler=train_sampler,
                                    train_batch_sampler=train_batch_sampler,
                                    test_sampler=test_sampler,
                                    test_batch_Sampler=test_batch_sampler,
                                    epoch=epoch,
                                    num_it_epoch=num_it_epoch,
                                    num_it_total=num_it_total,
                                    eval_epoch=eval_epoch,
                                    eval_it=eval_it,
                                    mu=mu,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    device=device,
                                    evaluation=evaluation
                                    )
        self.ema_decay=ema_decay
        self.lambda_u=lambda_u

        self.weight_decay=weight_decay
        self.warmup=warmup
        self.num_classes=num_classes
        self.num_samples=num_samples
        self.ema_pslab=None
        self.epoch_pslab=None

        if isinstance(self._augmentation,dict):
            self.weakly_augmentation=self._augmentation['augmentation']
            self.normalization = self._augmentation['normalization']
        elif isinstance(self._augmentation,(list,tuple)):
            self.weakly_augmentation = self._augmentation[0]
            self.normalization = self._augmentation[1]
        else:
            self.weakly_augmentation = copy.deepcopy(self._augmentation)
            self.normalization = copy.deepcopy(self._augmentation)

        if isinstance(self._optimizer,SemiOptimizer):
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in self._network.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self._network.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self._optimizer=self._optimizer.init_optimizer(params=grouped_parameters)

        if isinstance(self._scheduler,SemiLambdaLR):
            self._scheduler=self._scheduler.init_scheduler(optimizer=self._optimizer)

    def start_fit(self, *args, **kwargs):
        n_classes = self.num_classes if self.num_classes is not None else \
                        class_status(self._train_dataset.labled_dataset.y).num_class,
        n_samples = self.num_samples if self.num_samples is not None else \
                        self._train_dataset.unlabled_dataset.__len__()
        self.epoch_pslab = self.create_soft_pslab(n_samples=n_samples,
                                           n_classes=n_classes,dtype='rand')
        self.ema_pslab   = self.create_soft_pslab(n_samples=n_samples,
                                           n_classes=n_classes,dtype='zero')
    def start_epoch(self, *args, **kwargs):
        self._scheduler().step()
    def end_epoch(self, *args, **kwargs):
        self._optimizer.zero_grad()

        self._optimizer.step()

    def create_soft_pslab(self, n_samples, n_classes, dtype='rand'):
        if dtype == 'rand':
            pslab = torch.randint(0, n_classes, (n_samples, n_classes))
        elif dtype == 'zero':
            pslab = torch.zeros(n_samples, n_classes)
        else:
            raise ValueError('Unknown pslab dtype: {}'.format(dtype))
        return pslab.to(self.device)

    def update_ema_predictions(self):
        self.ema_pslab = (self.ema_decay*self.ema_pslab) + (1.0-self.ema_decay)*self.epoch_pslab
        self.epoch_pslab = self.ema_pslab / (1.0 - self.ema_decay**((self.epoch)+1.0))

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):

        lb_X=self.weakly_augmentation.fit_transform(copy.deepcopy(lb_X))
        ulb_X=self.weakly_augmentation.fit_transform(copy.deepcopy(ulb_X))

        self._network.apply(partial(fix_bn, train=True))
        logits_x_lb = self._network(lb_X)
        self._network.apply(partial(fix_bn,train=False))
        logits_x_ulb = self._network(ulb_X)
        iter_unlab_pslab = self.epoch_pslab[ulb_idx]
        with torch.no_grad():
            self.epoch_pslab[ulb_idx] = logits_x_ulb.clone().detach()

        return logits_x_lb,lb_y,logits_x_ulb,iter_unlab_pslab

    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb, lb_y, logits_x_ulb,iter_unlab_pslab  = train_result
        sup_loss = cross_entropy(logits_x_lb, lb_y, reduction='mean')
        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        unsup_loss = consistency_loss(logits_x_ulb.detach(),iter_unlab_pslab.detach())
        loss = sup_loss + _warmup * self.lambda_u *unsup_loss
        return loss


    def optimize(self,*args,**kwargs):
        self._optimizer.step()
        self._network.zero_grad()


    def estimate(self,X,idx=None,*args,**kwargs):
        X=self.normalization.fit_transform(X)
        outputs = self._network(X)
        return outputs

    def get_predict_result(self,y_est,*args,**kwargs):
        self.y_score=Softmax(dim=-1)(y_est)
        max_probs,y_pred=torch.max(self.y_score, dim=-1)
        return y_pred


    def predict(self,X=None):
        return SemiDeepModelMixin.predict(self,X=X)



