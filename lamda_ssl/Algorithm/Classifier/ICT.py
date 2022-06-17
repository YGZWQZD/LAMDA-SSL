from lamda_ssl.Base.InductiveEstimator import InductiveEstimator
from lamda_ssl.Base.SemiDeepModelMixin import SemiDeepModelMixin
from sklearn.base import ClassifierMixin

from lamda_ssl.utils import cross_entropy
import numpy as np

from lamda_ssl.utils import Bn_Controller
from lamda_ssl.Loss.Cross_Entropy import Cross_Entropy
import torch.nn as nn
import torch
from lamda_ssl.Transform.Mixup import Mixup

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
                 labeled_dataset=None,
                 unlabeled_dataset=None,
                 labeled_dataloader=None,
                 unlabeled_dataloader=None,
                 labeled_sampler=None,
                 unlabeled_sampler=None,
                 labeled_batch_sampler=None,
                 unlabeled_batch_sampler=None,
                 eval_epoch=None,
                 eval_it=None,
                 optimizer=None,
                 weight_decay=None,
                 scheduler=None,
                 device='cpu',
                 evaluation=None,
                 epoch=1,
                 num_it_epoch=None,
                 num_it_total=None,
                 mu=None,
                 ema_decay=None,
                 parallel=None,
                 file=None,
                 warmup=None,
                 lambda_u=None,
                 alpha=None):
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
                                    file=file,
                                    )
        self.ema_decay=ema_decay
        self.lambda_u=lambda_u
        self.weight_decay=weight_decay
        self.warmup=warmup
        self.alpha=alpha
        self.bn_controller=Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weakly_augmentation,dim=1,x=0,y=0)

    def start_fit(self):
        self.it_total = 0
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_x = lb_X[0] if isinstance(lb_X, (tuple, list)) else lb_X
        lb_y = lb_y[0] if isinstance(lb_y, (tuple, list)) else lb_y
        ulb_x_1 = ulb_X[0] if isinstance(ulb_X, (tuple, list)) else ulb_X
        logits_x_lb = self._network(lb_x)
        index = torch.randperm(ulb_x_1.size(0)).to(self.device)
        ulb_x_2=ulb_x_1[index]
        mixup=Mixup(self.alpha)

        if self.ema is not None:
            self.ema.apply_shadow()
        with torch.no_grad():
            logits_x_ulb_1 = self._network(ulb_x_1)
        if self.ema is not None:
            self.ema.restore()
        logits_x_ulb_2=logits_x_ulb_1[index]
        mixed_x= mixup.fit(ulb_x_1).transform(ulb_x_2)
        lam=mixup.lam
        self.bn_controller.freeze_bn(self._network)
        logits_x_ulb_mix = self._network(mixed_x)
        self.bn_controller.unfreeze_bn(self._network)
        return logits_x_lb,lb_y,logits_x_ulb_1,logits_x_ulb_2,logits_x_ulb_mix,lam

    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb,lb_y,logits_x_ulb_1,logits_x_ulb_2,logits_x_ulb_mix,lam=train_result
        sup_loss = cross_entropy(logits_x_lb, lb_y).mean()  # CE_loss for labeled data
        unsup_loss = Cross_Entropy(use_hard_labels=False, reduction='mean')(logits_x_ulb_mix,lam * nn.Softmax(dim=-1)(logits_x_ulb_1)+(1-lam)*
                                                                            nn.Softmax(dim=-1)(logits_x_ulb_2))
        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        loss = sup_loss + self.lambda_u * _warmup * unsup_loss
        return loss

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)



