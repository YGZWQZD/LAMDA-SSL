from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin

from sklearn.base import ClassifierMixin
from torch.nn import Softmax
from Semi_sklearn.utils import cross_entropy,consistency_loss
import numpy as np

import torch
from Semi_sklearn.Model.MeanTeacher import MeanTeacher

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

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
        MeanTeacher.__init__(self,train_dataset=train_dataset,
                             test_dataset=test_dataset,
                             train_dataloader=train_dataloader,
                             test_dataloader=test_dataloader,
                             augmentation=augmentation,
                             network=network,
                             train_sampler=train_sampler,
                             train_batch_sampler=train_batch_sampler,
                             test_sampler=test_sampler,
                             test_batch_sampler=test_batch_sampler,
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


    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb, lb_y, logits_x_ulb_1, logits_x_ulb_2=train_result
        sup_loss = cross_entropy(logits_x_lb, lb_y, reduction='mean')  # CE_loss for labeled data

        warm_up = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        unsup_loss = consistency_loss(logits_x_ulb_2, logits_x_ulb_1)  # MSE loss for unlabeled data
        loss = sup_loss + warm_up * self.lambda_u *unsup_loss
        return loss



    def get_predict_result(self,y_est,*args,**kwargs):

        self.y_score=Softmax(dim=-1)(y_est)
        max_probs,y_pred=torch.max(self.y_score, dim=-1)
        return y_pred

    def predict(self,X=None):
        return SemiDeepModelMixin.predict(self,X=X)



