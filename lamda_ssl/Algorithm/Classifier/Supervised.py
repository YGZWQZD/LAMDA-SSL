import copy
from lamda_ssl.Base.InductiveEstimator import InductiveEstimator
from lamda_ssl.Base.SemiDeepModelMixin import SemiDeepModelMixin
from lamda_ssl.Opitimizer.SemiOptimizer import SemiOptimizer
from lamda_ssl.Scheduler.SemiScheduler import SemiScheduler
from sklearn.base import ClassifierMixin
from lamda_ssl.utils import EMA
import torch
from lamda_ssl.utils import cross_entropy
from lamda_ssl.utils import class_status
from torch.nn import Softmax
import math

# def fix_bn(m,train=False):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         if train:
#             m.train()
#         else:
#             m.eval()

class Supervised(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
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
        self.threshold=threshold
        self.tsa_schedule=tsa_schedule
        self.num_classes=num_classes
        self.T=T
        self.weight_decay=weight_decay
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        # self._train_dataset.add_unlabeled_transform(copy.deepcopy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        # self._train_dataset.add_unlabeled_transform(self.strongly_augmentation,dim=1,x=1,y=0)
    def init_augmentation(self):
        # print(self._augmentation)
        if self._augmentation is not None:
            if isinstance(self._augmentation, dict):
                self.weakly_augmentation = self._augmentation['augmentation'] \
                    if 'augmentation' in self._augmentation.keys() \
                    else self._augmentation['weakly_augmentation']
                if 'strongly_augmentation' in self._augmentation.keys():
                    self.strongly_augmentation = self._augmentation['strongly_augmentation']
            elif isinstance(self._augmentation, (list, tuple)):
                self.weakly_augmentation = self._augmentation[0]
                if len(self._augmentation) > 1:
                    self.strongly_augmentation = self._augmentation[1]
            else:
                self.strongly_augmentation = copy.deepcopy(self._augmentation)
                self.weakly_augmentation=None

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_class
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X = lb_X[0] if isinstance(lb_X, (tuple, list)) else lb_X
        lb_y = lb_y[0] if isinstance(lb_y, (tuple, list)) else lb_y
        ulb_X=ulb_X[0] if isinstance(lb_y, (tuple, list)) else ulb_X
        num_lb = lb_X.shape[0]
        logits = self._network(lb_X)
        return logits,lb_y

    def get_loss(self,train_result,*args,**kwargs):
        logits,lb_y=train_result
        # sup_mask = torch.max(torch.softmax(logits_x_lb, dim=-1), dim=-1)[0].le(tsa).float().detach()
        sup_loss = (cross_entropy(logits, lb_y, reduction='none')).mean()  # CE_loss for labeled data
        return sup_loss

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)


