import copy
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from sklearn.base import RegressorMixin
import torch
from Semi_sklearn.utils import cross_entropy
from Semi_sklearn.Loss.Consistency import Consistency
import numpy as np
from Semi_sklearn.utils import class_status
from Semi_sklearn.utils import one_hot
from Semi_sklearn.Transform.Mixup import Mixup
import torch.nn.functional as F
from Semi_sklearn.utils import Bn_Controller
import torch.nn as nn
import  Semi_sklearn.Algorithm.ICT as ICT_model

class ICT(ICT_model.ICT,RegressorMixin):
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
                 scheduler=None,
                 device='cpu',
                 evaluation=None,
                 lambda_u=None,
                 mu=None,
                 ema_decay=None,
                 weight_decay=None,
                 # T=None,
                 # num_classes=10,
                 alpha=None,
                 file=None
                 ):
        ICT_model.ICT.__init__(self,train_dataset=train_dataset,
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
                                   warmup=warmup,
                                   lambda_u=lambda_u,
                                   alpha=alpha,
                               file=file
                                    )
        self._estimator_type = RegressorMixin._estimator_type


    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb,lb_y,logits_x_ulb_1,logits_x_ulb_2,logits_x_ulb_mix,lam=train_result
        sup_loss = Consistency(reduction='mean')(logits_x_lb, lb_y)  # CE_loss for labeled data
        unsup_loss = lam*Consistency(reduction='mean')(logits_x_ulb_mix,logits_x_ulb_1) +\
                     (1.0 - lam)*Consistency(reduction='mean')(logits_x_ulb_mix,logits_x_ulb_2)
        # unsup_loss=F.mse_loss(torch.softmax(logits_x_ulb, dim=-1), ulb_y, reduction='mean')
        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        print(unsup_loss)
        loss = sup_loss + self.lambda_u * _warmup * unsup_loss
        return loss

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)



