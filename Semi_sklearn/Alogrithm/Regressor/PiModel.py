from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin

from sklearn.base import RegressorMixin
from Semi_sklearn.Loss.Consistency import Consistency
import numpy as np

from Semi_sklearn.Alogrithm.Pimodel import PiModel

class PiModelClassifier(PiModel,RegressorMixin):
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
                 warmup=0.4,
                 eval_epoch=None,
                 eval_it=None,
                 optimizer=None,
                 scheduler=None,
                 device='cpu',
                 evaluation=None,
                 lambda_u=None,
                 mu=None,
                 ema_decay=None,
                 weight_decay=None
                 ):
        PiModel.__init__(self,train_dataset=train_dataset,
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
                             warmup=warmup,
                             eval_epoch=eval_epoch,
                             eval_it=eval_it,
                             optimizer=optimizer,
                             scheduler=scheduler,
                             device=device,
                             evaluation=evaluation,
                             lambda_u=lambda_u,
                             mu=mu,
                             ema_decay=ema_decay,
                             weight_decay=weight_decay)
        self._estimator_type = RegressorMixin._estimator_type


    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb, lb_y, logits_x_ulb_1, logits_x_ulb_2=train_result
        sup_loss = Consistency()(logits_x_lb, lb_y)
        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        unsup_loss = Consistency()(logits_x_ulb_1,logits_x_ulb_2.detach())
        loss = sup_loss + _warmup * self.lambda_u *unsup_loss
        return loss

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)



