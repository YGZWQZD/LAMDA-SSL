from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from TwoAdaption.Deep.Config.Default_Config import config
from LAMDA_SSL.utils import Bn_Controller
from sklearn.base import ClassifierMixin
from LAMDA_SSL.utils import to_device
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from LAMDA_SSL.Base.BaseOptimizer import BaseOptimizer
from LAMDA_SSL.Base.BaseScheduler import BaseScheduler
from LAMDA_SSL.utils import class_status

class UASD(InductiveEstimator,DeepModelMixin):
    def __init__(self,
                 lambda_u=1.0,
                 ema_weight=None,
                 warmup=None,
                 num_classes=None,
                 num_samples = None,
                 mu=1.0,
                 ema_decay=None,
                 threshold=0.95,
                 weight_decay=None,
                 epoch=1,
                 num_it_epoch=2000,
                 num_it_total=None,
                 eval_epoch=None,
                 eval_it=None,
                 device='cuda:0',
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
                 verbose=config.verbose
                 ):
        # >> Parameter
        # >> - lambda_u: The weight of unsupervised loss.
        # >> - warmup: The end position of warmup. For example, num_it_total is 100 and warmup is 0.4, then warmup is performed in the first 40 iterations.
        # >> - ema_weight: Update weight for exponential moving average pseudo labels。
        # >> - num_classes: The number of classes.
        # >> - num_samples: The number of samples.
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
        self.ema_weight=ema_weight
        self.lambda_u=lambda_u
        self.weight_decay=weight_decay
        self.warmup=warmup
        self.threshold=threshold
        self.num_classes=num_classes
        self.num_samples=num_samples
        self.ema_pslab=None
        self.epoch_pslab=None
        self.bn_controller=Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def start_fit(self):
        self.init_epoch()
        num_classes = self.num_classes if self.num_classes is not None else \
                        class_status(self._train_dataset.labeled_dataset.y).num_classes
        print(num_classes)
        num_samples = self.num_samples if self.num_samples is not None else \
                        self._train_dataset.unlabeled_dataset.__len__()
        self.epoch_pslab = torch.zeros(num_samples, num_classes).to(self.device)
        self.pslab   = torch.zeros(num_samples, num_classes).to(self.device)
        self.it_total = 0
        self._network.zero_grad()
        self._network.train()

    def end_fit_epoch(self):
        self.update_predictions()

    def update_predictions(self):
        self.pslab = ((self._epoch-1)*self.pslab + self.epoch_pslab)/self._epoch


    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):

        lb_X = lb_X[0] if isinstance(lb_X, (tuple, list)) else lb_X
        lb_y = lb_y[0] if isinstance(lb_y, (tuple, list)) else lb_y
        ulb_X = ulb_X[0] if isinstance(ulb_X, (tuple, list)) else ulb_X
        lb_logits = self._network(lb_X)
        self.bn_controller.freeze_bn(self._network)
        ulb_logits = self._network(ulb_X)
        self.bn_controller.unfreeze_bn(self._network)
        iter_unlab_pslab = self.pslab[ulb_idx]
        with torch.no_grad():
            self.epoch_pslab[ulb_idx] = ulb_logits.softmax(1).clone().detach()

        return lb_logits,lb_y,ulb_logits,iter_unlab_pslab


    def get_loss(self,train_result,*args,**kwargs):
        lb_logits,lb_y,ulb_logits,iter_unlab_pslab  = train_result
        sup_loss = CrossEntropyLoss(reduction='mean')(lb_logits, lb_y)
        iter_unlab_pslab=(iter_unlab_pslab*(self._epoch-1)+ulb_logits.softmax(1))/self._epoch
        max_probs, targets_u = torch.max(iter_unlab_pslab, dim=-1)
        mask = max_probs.ge(self.threshold).float()
        unsup_loss = (CrossEntropyLoss(reduction='none')(ulb_logits, iter_unlab_pslab.detach()) * mask).mean()
        loss=sup_loss+self.lambda_u*(self._epoch/self.epoch)*unsup_loss
        return loss

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)