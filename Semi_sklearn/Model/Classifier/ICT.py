import copy
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from sklearn.base import ClassifierMixin
import torch
from Semi_sklearn.utils import cross_entropy
import numpy as np
from Semi_sklearn.utils import class_status
from Semi_sklearn.utils import one_hot
from Semi_sklearn.Transform.Mixup import Mixup
import torch.nn.functional as F
from Semi_sklearn.utils import Bn_Controller
import torch.nn as nn

class ICT(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
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
                 T=None,
                 num_classes=10,
                 alpha=None
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
                                    evaluation=evaluation
                                    )
        self.ema_decay=ema_decay
        self.lambda_u=lambda_u
        self.weight_decay=weight_decay
        self.warmup=warmup
        self.T=T
        self.alpha=alpha
        self.num_classes=num_classes
        self.bn_controller=Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabled_transform(self.weakly_augmentation,dim=1,x=0,y=0)

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labled_dataset.y).num_class
        self.it_total = 0
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):

        lb_x = lb_X[0]
        ulb_x_1 =ulb_X[0]
        logits_x_lb = self._network(lb_x)
        index = torch.randperm(ulb_x_1.size(0)).to(self.device)
        ulb_x_2=ulb_x_1[index]
        mixup=Mixup(self.alpha)
        mixed_x = mixup.fit(ulb_x_1).transform(ulb_x_2)
        lam=mixup.lam

        self.bn_controller.freeze_bn(self._network)
        logits_x_ulb_1 = self._network(ulb_x_1)
        logits_x_ulb_2 = self._network(ulb_x_2)
        logits_x_ulb_mix = self._network(mixed_x)
        self.bn_controller.unfreeze_bn(self._network)

        return logits_x_lb,lb_y,logits_x_ulb_1,logits_x_ulb_2,logits_x_ulb_mix,lam

    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb,lb_y,logits_x_ulb_1,logits_x_ulb_2,logits_x_ulb_mix,lam=train_result
        sup_loss = cross_entropy(logits_x_lb, lb_y).mean()  # CE_loss for labeled data
        unsup_loss = lam*nn.CrossEntropyLoss()(logits_x_ulb_mix,logits_x_ulb_1) +\
                     (1.0 - lam)*nn.CrossEntropyLoss()(logits_x_ulb_mix,logits_x_ulb_1)
        # unsup_loss=F.mse_loss(torch.softmax(logits_x_ulb, dim=-1), ulb_y, reduction='mean')
        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        loss = sup_loss + self.lambda_u * _warmup * unsup_loss
        return loss

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)



