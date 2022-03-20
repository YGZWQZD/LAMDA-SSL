import copy
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from Semi_sklearn.Opitimizer.SemiOptimizer import SemiOptimizer
from Semi_sklearn.Scheduler.SemiScheduler import SemiScheduler
from sklearn.base import RegressorMixin
from torch.nn import Softmax
import torch
from Semi_sklearn.utils import partial
from Semi_sklearn.utils import class_status
from Semi_sklearn.utils import cross_entropy,consistency_loss
from Semi_sklearn.utils import Bn_Controller
import Semi_sklearn.Model.TemporalEnsembling as Base_TemporalEnsembling
from Semi_sklearn.Loss.Consistency import Consistency
import numpy as np

# def fix_bn(m,train=False):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         if train:
#             m.train()
#         else:
#             m.eval()

class TemporalEnsembling(Base_TemporalEnsembling.TemporalEnsembling,RegressorMixin):
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
                 num_classes=None,
                 num_samples=None
                 ):
        Base_TemporalEnsembling.TemporalEnsembling.__init__(self,train_dataset=train_dataset,
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
                                    num_classes=num_classes,
                                    num_samples=num_samples
                                    )
        self._estimator_type = RegressorMixin._estimator_type


    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb, lb_y, logits_x_ulb,iter_unlab_pslab  = train_result
        sup_loss = Consistency(reduction='mean')(logits_x_lb, lb_y)
        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        unsup_loss = Consistency(reduction='mean')(logits_x_ulb,iter_unlab_pslab.detach())
        loss = sup_loss + _warmup * self.lambda_u *unsup_loss
        return loss

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)



