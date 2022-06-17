import copy
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from Semi_sklearn.Opitimizer.SemiOptimizer import SemiOptimizer
from Semi_sklearn.Scheduler.SemiScheduler import SemiScheduler
from sklearn.base import ClassifierMixin
from Semi_sklearn.utils import EMA
import torch
from Semi_sklearn.utils import cross_entropy
from Semi_sklearn.utils import partial
import numpy as np
from torch.nn import Softmax
import torch.nn as nn
from Semi_sklearn.utils import class_status
from Semi_sklearn.utils import one_hot
from Semi_sklearn.Transform.Mixup import Mixup
import torch.nn.functional as F
from Semi_sklearn.Transform.Rotate import Rotate
from Semi_sklearn.utils import Bn_Controller



class S4L(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
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
                 weight_decay=None,
                 scheduler=None,
                 device='cpu',
                 mu=None,
                 ema_decay=None,
                 evaluation=None,
                 parallel=None,
                 file=None,
                 lambda_u=None,
                 num_classes=10,
                 p_target=None,
                 rotate_v_list=None,
                 labeled_usp=True,
                 all_rot=True
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
                                    evaluation=evaluation,
                                    parallel=parallel,
                                    file=file
                                    )
        self.ema_decay=ema_decay
        self.lambda_u=lambda_u
        self.weight_decay=weight_decay
        self.num_classes=num_classes
        self.rotate_v_list=rotate_v_list if rotate_v_list is not None else [0, 90, 180, 270]
        self.p_model = None
        self.p_target=p_target
        self.labeled_usp=labeled_usp
        self.all_rot=all_rot
        self.bn_controller = Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weakly_augmentation,dim=1,x=0,y=0)
    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_classes
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):

        lb_x_w=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y
        ulb_x_w = ulb_X[0] if isinstance(ulb_X, (tuple, list)) else ulb_X

        logits_x_lb_w = self._network(lb_x_w)[0]

        # print(logits_x_lb_w.shape)
        # print(lb_y.shape)
        rot_x = torch.Tensor().to(self.device)
        rot_y = []

        for item in ulb_x_w:
            if self.all_rot:
                for _v in self.rotate_v_list:
                    rot_x = torch.cat((rot_x, Rotate(v=_v).fit_transform(item).unsqueeze(0)), dim=0)
                    rot_y.append(self.rotate_v_list.index(_v))
            else:
                _v = np.random.choice(self.rotate_v_list, 1).item()
                rot_x = torch.cat((rot_x, Rotate(v=_v).fit_transform(item).unsqueeze(0)), dim=0)
                rot_y.append(self.rotate_v_list.index(_v))
        if self.labeled_usp:
            for item in lb_x_w:
                if self.all_rot:
                    for _v in self.rotate_v_list:
                        rot_x = torch.cat((rot_x, Rotate(v=_v).fit_transform(item).unsqueeze(0)), dim=0)
                        rot_y.append(self.rotate_v_list.index(_v))
                else:
                    _v = np.random.choice(self.rotate_v_list, 1).item()
                    rot_x = torch.cat((rot_x, Rotate(v=_v).fit_transform(item).unsqueeze(0)), dim=0)
                    rot_y.append(self.rotate_v_list.index(_v))

        rot_y = torch.LongTensor(rot_y).to(self.device)

        # self.bn_controller.freeze_bn(model=self._network)
        logits_x_rot = self._network(rot_x)[1]
        # self.bn_controller.unfreeze_bn(model=self._network)
        # print(torch.any(torch.isnan(logits_x_ulb_w)))
        # prob_x_ulb = torch.softmax(logits_x_ulb_w, dim=1)

        return logits_x_lb_w,lb_y,logits_x_rot,rot_y


    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb_w,lb_y,logits_x_rot,rot_y=train_result
        sup_loss = cross_entropy(logits_x_lb_w, lb_y,use_hard_labels=True).mean()  # CE_loss for labeled data
        # print(rot_y.dtype)
        rot_loss = cross_entropy(logits_x_rot, rot_y, reduction='mean').mean()
        # _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        loss = sup_loss +self.lambda_u*rot_loss
        return loss

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)



