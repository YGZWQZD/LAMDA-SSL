import copy
from lamda_ssl.Base.InductiveEstimator import InductiveEstimator
from lamda_ssl.Base.SemiDeepModelMixin import SemiDeepModelMixin
from lamda_ssl.Opitimizer.SemiOptimizer import SemiOptimizer
from lamda_ssl.Scheduler.SemiScheduler import SemiScheduler
from sklearn.base import ClassifierMixin
from lamda_ssl.utils import EMA
import torch
from lamda_ssl.utils import cross_entropy
from lamda_ssl.utils import partial
import numpy as np
from torch.nn import Softmax
from lamda_ssl.utils import Bn_Controller

class PseudoLabel(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
    def __init__(self,train_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 train_dataloader=None,
                 valid_dataloader=None,
                 test_dataloader=None,
                 augmentation=None,
                 network=None,
                 train_sampler=None,
                 train_batch_sampler=None,
                 valid_sampler=None,
                 valid_batch_sampler=None,
                 test_sampler=None,
                 test_batch_sampler=None,
                 labeled_dataset=None,
                 unlabeled_dataset=None,
                 labeled_dataloader=None,
                 unlabeled_dataloader=None,
                 labeled_sampler=None,
                 unlabeled_sampler=None,
                 labeled_batch_sampler=None,
                 unlabeled_batch_sampler=None,
                 epoch=1,
                 num_it_epoch=None,
                 num_it_total=None,
                 warmup=None,
                 eval_epoch=None,
                 eval_it=None,
                 optimizer=None,
                 weight_decay=None,
                 scheduler=None,
                 device='cpu',
                 mu=None,
                 ema_decay=None,
                 evaluation=None,
                 parallel=None,
                 file=None,
                 lambda_u=None,
                 threshold=0.95
                 ):
        SemiDeepModelMixin.__init__(self,train_dataset=train_dataset,
                                    valid_dataset=valid_dataset,
                                    test_dataset=test_dataset,
                                    train_dataloader=train_dataloader,
                                    valid_dataloader=valid_dataloader,
                                    test_dataloader=test_dataloader,
                                    augmentation=augmentation,
                                    network=network,
                                    train_sampler=train_sampler,
                                    train_batch_sampler=train_batch_sampler,
                                    valid_sampler=valid_sampler,
                                    valid_batch_sampler=valid_batch_sampler,
                                    test_sampler=test_sampler,
                                    test_batch_sampler=test_batch_sampler,
                                    labeled_dataset=labeled_dataset,
                                    unlabeled_dataset=unlabeled_dataset,
                                    labeled_dataloader=labeled_dataloader,
                                    unlabeled_dataloader=unlabeled_dataloader,
                                    labeled_sampler=labeled_sampler,
                                    unlabeled_sampler=unlabeled_sampler,
                                    labeled_batch_sampler=labeled_batch_sampler,
                                    unlabeled_batch_sampler=unlabeled_batch_sampler,
                                    epoch=epoch,
                                    num_it_epoch=num_it_epoch,
                                    num_it_total=num_it_total,
                                    eval_epoch=eval_epoch,
                                    eval_it=eval_it,
                                    mu=mu,
                                    weight_decay=weight_decay,
                                    ema_decay=ema_decay,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    device=device,
                                    evaluation=evaluation,
                                    parallel=parallel,
                                    file=file
                                    )
        self.ema_decay=ema_decay
        self.lambda_u=lambda_u
        self.threshold=threshold
        self.weight_decay=weight_decay
        self.warmup=warmup
        self.bn_controller=Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):

        w_lb_X=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y
        w_ulb_X = ulb_X[0] if isinstance(ulb_X, (tuple, list)) else ulb_X

        logits_x_lb = self._network(w_lb_X)

        self.bn_controller.freeze_bn(self._network)
        logits_x_ulb = self._network(w_ulb_X)
        self.bn_controller.unfreeze_bn(self._network)

        return logits_x_lb,lb_y,logits_x_ulb


    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb,lb_y,logits_x_ulb=train_result
        sup_loss = cross_entropy(logits_x_lb, lb_y, reduction='mean')  # CE_loss for labeled data

        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        pseudo_label = torch.softmax(logits_x_ulb, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()

        unsup_loss = (cross_entropy(logits_x_ulb, max_idx.detach())*mask ).mean() # MSE loss for unlabeled data

        loss = sup_loss + self.lambda_u * unsup_loss * _warmup
        return loss

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)


