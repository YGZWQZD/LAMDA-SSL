import copy
from lamda_ssl.Base.InductiveEstimator import InductiveEstimator
from lamda_ssl.Base.DeepModelMixin import DeepModelMixin

from sklearn.base import ClassifierMixin
import numpy as np
import torch
import lamda_ssl.Config.VAT as config
from lamda_ssl.utils import _l2_normalize,kl_div_with_logit,cross_entropy
from torch.autograd import Variable
import torch.nn.functional as F
from lamda_ssl.utils import Bn_Controller

class VAT(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 lambda_u=config.lambda_u,
                 eps=config.eps,
                 warmup=config.warmup,
                 it_vat=config.it_vat,
                 xi=config.xi,
                 lambda_entmin=config.lambda_entmin,
                 mu=config.mu,
                 ema_decay=config.ema_decay,
                 weight_decay=config.weight_decay,
                 epoch=config.epoch,
                 num_it_epoch=config.num_it_epoch,
                 num_it_total=config.num_it_total,
                 eval_epoch=config.eval_epoch,
                 eval_it=config.eval_it,
                 device=config.device,
                 train_dataset=config.train_dataset,
                 labeled_dataset=config.labeled_dataset,
                 unlabeled_dataset=config.unlabeled_dataset,
                 valid_dataset=config.valid_dataset,
                 test_dataset=config.test_dataset,
                 train_dataloader=config.train_dataloader,
                 labeled_dataloader=config.labeled_dataloader,
                 unlabeled_dataloader=config.unlabeled_dataloader,
                 valid_dataloader=config.valid_dataloader,
                 test_dataloader=config.test_dataloader,
                 train_sampler=config.train_sampler,
                 train_batch_sampler=config.train_batch_sampler,
                 valid_sampler=config.valid_sampler,
                 valid_batch_sampler=config.valid_batch_sampler,
                 test_sampler=config.test_sampler,
                 test_batch_sampler=config.test_batch_sampler,
                 labeled_sampler=config.labeled_sampler,
                 unlabeled_sampler=config.unlabeled_sampler,
                 labeled_batch_sampler=config.labeled_batch_sampler,
                 unlabeled_batch_sampler=config.unlabeled_batch_sampler,
                 augmentation=config.augmentation,
                 network=config.network,
                 optimizer=config.optimizer,
                 scheduler=config.scheduler,
                 evaluation=config.evaluation,
                 parallel=config.parallel,
                 file=config.file,
                 verbose=config.verbose):
        DeepModelMixin.__init__(self,train_dataset=train_dataset,
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
                                    verbose=verbose
                                    )
        self.ema_decay=ema_decay
        self.lambda_u=lambda_u
        self.weight_decay=weight_decay
        self.warmup=warmup
        self.eps=eps
        self.it_vat=it_vat
        self.xi=xi
        self.lambda_entmin=lambda_entmin
        self.bn_controller=Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type


    def start_fit(self):
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        _lb_X=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y
        _ulb_X = ulb_X[0] if isinstance(ulb_X, (tuple, list)) else ulb_X

        logits_x_lb = self._network(_lb_X)
        # print(torch.any(torch.isnan(logits_x_lb)))
        self.bn_controller.freeze_bn(self._network)
        logits_x_ulb = self._network(_ulb_X)

        # print(torch.any(torch.isnan(logits_x_lb)))
        # print(torch.any(torch.isnan(logits_x_ulb)))
        d = torch.Tensor(_ulb_X.size()).normal_()
        for i in range(self.it_vat):
            d = self.xi*_l2_normalize(d)# r=self.xi*_l2_normalize(d)
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
        return logits_x_lb,lb_y,logits_x_ulb, y_hat

    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb,lb_y,logits_x_ulb,y_hat=train_result
        unsup_warmup = np.clip(self.it_total / (self.warmup * self.num_it_total),
                a_min=0.0, a_max=1.0)

        sup_loss = cross_entropy(logits_x_lb, lb_y, reduction='mean')
        # print(sup_loss)
        unsup_loss = kl_div_with_logit(logits_x_ulb.detach(), y_hat)
        # print(unsup_loss)
        p = F.softmax(logits_x_ulb, dim=1)
        entmin_loss=-(p*F.log_softmax(logits_x_ulb, dim=1)).sum(dim=1).mean(dim=0)
        # print(entmin_loss)
        loss = sup_loss + self.lambda_u * unsup_loss * unsup_warmup + self.lambda_entmin * entmin_loss
        return loss

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)


