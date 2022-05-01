import copy
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from Semi_sklearn.Opitimizer.SemiOptimizer import SemiOptimizer
from Semi_sklearn.Scheduler.SemiScheduler import SemiScheduler
from sklearn.base import ClassifierMixin
from torch.nn import Softmax
import torch
from Semi_sklearn.utils import partial
from Semi_sklearn.utils import class_status
from Semi_sklearn.utils import cross_entropy,consistency_loss
from Semi_sklearn.utils import Bn_Controller
import numpy as np

# def fix_bn(m,train=False):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         if train:
#             m.train()
#         else:
#             m.eval()

class TemporalEnsembling(InductiveEstimator,SemiDeepModelMixin):
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
                 ema_weight=None,
                 ema_decay=None,
                 weight_decay=None,
                 num_classes=None,
                 num_samples=None
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
        self.ema_weight=ema_weight
        self.lambda_u=lambda_u
        self.weight_decay=weight_decay
        self.warmup=warmup
        self.num_classes=num_classes
        self.num_samples=num_samples
        self.ema_pslab=None
        self.epoch_pslab=None
        self.bn_controller=Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def start_fit(self):
        n_classes = self.num_classes if self.num_classes is not None else \
                        class_status(self._train_dataset.labeled_dataset.y).num_class

        n_samples = self.num_samples if self.num_samples is not None else \
                        self._train_dataset.unlabeled_dataset.__len__()
        # self.epoch_pslab = self.create_soft_pslab(n_samples=n_samples,
        #                                    n_classes=n_classes,dtype='rand')
        self.ema_pslab   = self.create_soft_pslab(n_samples=n_samples,
                                           n_classes=n_classes,dtype='zero')
        self.it_total = 0
        self._network.zero_grad()
        self._network.train()

    def end_epoch(self):
        if self._scheduler is not None:
                self._scheduler.step()

    def create_soft_pslab(self, n_samples, n_classes, dtype='rand'):
        if dtype == 'rand':
            pslab = torch.randint(0, n_classes, (n_samples, n_classes))
        elif dtype == 'zero':
            pslab = torch.zeros(n_samples, n_classes)
        else:
            raise ValueError('Unknown pslab dtype: {}'.format(dtype))
        return pslab.to(self.device)

    def update_ema_predictions(self,iter_pslab,idxs):
        ema_iter_pslab = (self.ema_weight * self.ema_pslab[idxs]) + (1.0 - self.ema_weight) * iter_pslab
        self.ema_pslab[idxs] = ema_iter_pslab
        return ema_iter_pslab / (1.0 - self.ema_weight ** (self._epoch+ 1.0))

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):


        # _ulb_idx=ulb_idx.tolist() if ulb_idx is not None else ulb_idx

        lb_X = lb_X[0] if isinstance(lb_X, (tuple, list)) else lb_X
        lb_y = lb_y[0] if isinstance(lb_y, (tuple, list)) else lb_y
        ulb_X = ulb_X[0] if isinstance(ulb_X, (tuple, list)) else ulb_X
        logits_x_lb = self._network(lb_X)
        self.bn_controller.freeze_bn(self._network)
        logits_x_ulb = self._network(ulb_X)
        self.bn_controller.unfreeze_bn(self._network)
        # with torch.no_grad():
        #     print(self.epoch_pslab.dtype)
        #     iter_unlab_pslab = self.update_ema_predictions(logits_x_ulb.clone().detach(),ulb_X)
        with torch.no_grad():
            iter_unlab_pslab = self.update_ema_predictions(logits_x_ulb.clone().detach(),ulb_idx)

        return logits_x_lb,lb_y,logits_x_ulb,iter_unlab_pslab
        # return logits_x_lb,lb_y

    def optimize(self,loss,*args,**kwargs):
        self._network.zero_grad()
        loss.backward()
        self._optimizer.step()
        if self.ema is not None:
            self.ema.update()
        # if self._scheduler is not None:
        #     self._scheduler.step()
    # # def optimize(self,loss,*args,**kwargs):
    #     self._network.zero_grad()
    #     loss.backward()
    #     self._optimizer.step()
    #     if self._scheduler is not None:
    #         self._scheduler.step()





    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb, lb_y, logits_x_ulb,iter_unlab_pslab  = train_result
        # logits_x_lb, lb_y=train_result
        # print(logits_x_lb[0])
        # print(lb_y[0])
        sup_loss = cross_entropy(logits_x_lb, lb_y, reduction='mean')
        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        unsup_loss = consistency_loss(logits_x_ulb,iter_unlab_pslab.detach())
        loss = sup_loss + _warmup * self.lambda_u *unsup_loss
        return loss
        # return sup_loss
    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)



