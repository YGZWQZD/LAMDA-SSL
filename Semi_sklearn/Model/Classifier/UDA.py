import copy
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from Semi_sklearn.Opitimizer.SemiOptimizer import SemiOptimizer
from Semi_sklearn.Scheduler.SemiScheduler import SemiScheduler
from sklearn.base import ClassifierMixin
from Semi_sklearn.utils import EMA
import torch
from Semi_sklearn.utils import cross_entropy
from Semi_sklearn.utils import class_status
from torch.nn import Softmax
import math

# def fix_bn(m,train=False):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         if train:
#             m.train()
#         else:
#             m.eval()

class UDA(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
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
                 eval_epoch=None,
                 eval_it=None,
                 optimizer=None,
                 scheduler=None,
                 device='cpu',
                 evaluation=None,
                 lambda_u=None,
                 mu=None,
                 ema_decay=None,
                 threshold=0.95,
                 num_classes=None,
                 tsa_schedule=None,
                 weight_decay=None,
                 T=0.4
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
        self.threshold=threshold
        self.tsa_schedule=tsa_schedule
        self.num_classes=num_classes
        self.T=T
        self.weight_decay=weight_decay
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_unlabled_transform(copy.deepcopy(self.train_dataset.unlabled_transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabled_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabled_transform(self.strongly_augmentation,dim=1,x=1,y=0)

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labled_dataset.y).num_class
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X=lb_X[0]
        w_ulb_X,s_ulb_X=ulb_X[0],ulb_X[1]
        num_lb = lb_X.shape[0]
        inputs = torch.cat([lb_X, w_ulb_X, s_ulb_X], dim=0)
        logits = self._network(inputs)
        logits_x_lb = logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
        return logits_x_lb,lb_y,logits_x_ulb_w, logits_x_ulb_s

    def get_tsa(self):
        training_progress = self.it_total / self.num_it_total

        if self.tsa_schedule is None or self.tsa_schedule == 'none':
            return 1
        else:
            if self.tsa_schedule == 'linear':
                threshold = training_progress
            elif self.tsa_schedule == 'exp':
                scale = 5
                threshold = math.exp((training_progress - 1) * scale)
            elif self.tsa_schedule == 'log':
                scale = 5
                threshold = 1 - math.exp((-training_progress) * scale)
            else:
                raise ValueError('Can not get tsa' )
            tsa = threshold * (1 - 1 / self.num_classes) + 1 / self.num_classes
            return tsa

    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb,lb_y,logits_x_ulb_w, logits_x_ulb_s=train_result
        tsa = self.get_tsa()
        sup_mask = torch.max(torch.softmax(logits_x_lb, dim=-1), dim=-1)[0].le(tsa).float().detach()
        sup_loss = (cross_entropy(logits_x_lb, lb_y, reduction='none')* sup_mask).mean()  # CE_loss for labeled data
        pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()
        pseudo_label = torch.softmax(logits_x_ulb_w / self.T, dim=-1)
        unsup_loss = (cross_entropy(logits_x_ulb_s, pseudo_label,use_hard_labels=False)*mask ).mean() # MSE loss for unlabeled data
        loss = sup_loss + self.lambda_u * unsup_loss
        return loss

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)


