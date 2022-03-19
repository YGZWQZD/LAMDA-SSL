import copy
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin

from sklearn.base import ClassifierMixin
import numpy as np
from Semi_sklearn.utils import EMA
import torch
from Semi_sklearn.utils import class_status
from Semi_sklearn.utils import partial
from torch.nn import Softmax
from Semi_sklearn.utils import _l2_normalize,kl_div_with_logit,cross_entropy
from torch.autograd import Variable
import torch.nn.functional as F
from Semi_sklearn.utils import Bn_Controller

# def fix_bn(m,train=False):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         if train:
#             m.train()
#         else:
#             m.eval()

class VAT(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
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
                 eval_epoch=None,
                 eval_it=None,
                 optimizer=None,
                 scheduler=None,
                 device='cpu',
                 evaluation=None,
                 lambda_u=None,
                 mu=None,
                 ema_decay=None,
                 num_classes=None,
                 tsa_schedule=None,
                 weight_decay=None,
                 eps=6,
                 warmup=None,
                 it_vat=1,
                 xi=1e-6,
                 lambda_entmin=0.06
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
                                    evaluation=evaluation
                                    )
        self.ema_decay=ema_decay
        self.lambda_u=lambda_u
        self.tsa_schedule=tsa_schedule
        self.num_classes=num_classes
        self.weight_decay=weight_decay
        self.warmup=warmup
        self.eps=eps
        self.it_vat=it_vat
        self.xi=xi
        self.lambda_entmin=lambda_entmin
        self.bn_controller=Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type


    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_class
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        _lb_X=lb_X[0]
        _ulb_X=ulb_X[0]

        logits_x_lb = self._network(_lb_X)
        # print(torch.any(torch.isnan(logits_x_lb)))
        self.bn_controller.freeze_bn(self._network)
        logits_x_ulb = self._network(_ulb_X)

        # print(torch.any(torch.isnan(logits_x_lb)))
        # print(torch.any(torch.isnan(logits_x_ulb)))
        d = torch.Tensor(_ulb_X.size()).normal_()
        for i in range(self.it_vat):
            d = _l2_normalize(d)
            d = Variable(d.to(self.device), requires_grad=True)
            y_hat = self._network(_ulb_X + d)
            # print(torch.any(torch.isnan(y_hat)))
            delta_kl = kl_div_with_logit(logits_x_ulb.detach(), y_hat)
            # print(delta_kl)
            delta_kl.backward()
            d = d.grad.data.clone()
            self._network.zero_grad()

        d = _l2_normalize(d)
        d = Variable(d)
        r_adv = self.eps * d
        y_hat = self._network(_ulb_X + r_adv.detach())
        # print(torch.any(torch.isnan(y_hat)))
        self.bn_controller.unfreeze_bn(self._network)
        logits_x_ulb=logits_x_ulb.detach()
        return logits_x_lb,lb_y,logits_x_ulb, y_hat

    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb,lb_y,logits_x_ulb,y_hat=train_result
        unsup_warmup = np.clip(self.it_total / (self.warmup * self.num_it_total),
                a_min=0.0, a_max=1.0)

        sup_loss = cross_entropy(logits_x_lb, lb_y, reduction='mean')
        # print(sup_loss)
        unsup_loss = kl_div_with_logit(logits_x_ulb, y_hat)
        # print(unsup_loss)
        p = F.softmax(logits_x_ulb, dim=1)
        entmin_loss=-(p*F.log_softmax(logits_x_ulb, dim=1)).sum(dim=1).mean(dim=0)
        # print(entmin_loss)
        loss = sup_loss + self.lambda_u * unsup_loss * unsup_warmup + self.lambda_entmin * entmin_loss
        return loss

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)


