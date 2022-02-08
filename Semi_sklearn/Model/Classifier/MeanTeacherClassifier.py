from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin

from sklearn.base import ClassifierMixin
from Semi_sklearn.utils import cross_entropy,consistency_loss
import numpy as np

from Semi_sklearn.Model.MeanTeacher import MeanTeacher

class MeanTeacherClassifier(MeanTeacher,ClassifierMixin):
    def __init__(self,train_dataset=None,
                 test_dataset=None,
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
                 weight_decay=None
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
                                    weight_decay=weight_decay,
                                    ema_decay=ema_decay,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    device=device,
                                    evaluation=evaluation)
        self._estimator_type = ClassifierMixin._estimator_type


    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb, lb_y, logits_x_ulb_1, logits_x_ulb_2=train_result
        sup_loss = cross_entropy(logits_x_lb, lb_y, reduction='mean')  # CE_loss for labeled data

        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        unsup_loss = consistency_loss(logits_x_ulb_2, logits_x_ulb_1.detach())  # MSE loss for unlabeled data
        loss = sup_loss + _warmup * self.lambda_u *unsup_loss
        return loss

    def predict(self,X=None):
        return SemiDeepModelMixin.predict(self,X=X)



