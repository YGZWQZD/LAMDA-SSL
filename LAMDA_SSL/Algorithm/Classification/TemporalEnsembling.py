from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
import torch
from LAMDA_SSL.utils import class_status
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
from LAMDA_SSL.Loss.Consistency import Consistency
from LAMDA_SSL.Loss.Semi_Supervised_Loss import Semi_Supervised_Loss
from LAMDA_SSL.utils import Bn_Controller
import numpy as np
import LAMDA_SSL.Config.TemporalEnsembling as config

class TemporalEnsembling(InductiveEstimator,DeepModelMixin):
    def __init__(self,
                 lambda_u=config.lambda_u,
                 ema_weight=config.ema_weight,
                 warmup=config.warmup,
                 num_classes=config.num_classes,
                 num_samples = config.num_samples,
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

        num_samples = self.num_samples if self.num_samples is not None else \
                        self._train_dataset.unlabeled_dataset.__len__()
        self.epoch_pslab = self.create_soft_pslab(num_samples=num_samples,
                                           num_classes=num_classes,dtype='rand')
        self.ema_pslab   = self.create_soft_pslab(num_samples=num_samples,
                                           num_classes=num_classes,dtype='zero')
        self.it_total = 0
        self._network.zero_grad()
        self._network.train()

    def end_fit_epoch(self):
        self.update_ema_predictions()
        if self._scheduler is not None:
                self._scheduler.step()

    def create_soft_pslab(self, num_samples, num_classes, dtype='rand'):
        if dtype == 'rand':
            pslab = torch.randint(0, num_classes, (num_samples, num_classes)).float()
        elif dtype == 'zero':
            pslab = torch.zeros(num_samples, num_classes)
        else:
            raise ValueError('Unknown pslab dtype: {}'.format(dtype))
        return pslab.to(self.device)

    def update_ema_predictions(self):
        self.ema_pslab = (self.ema_weight * self.ema_pslab) + (1.0 - self.ema_weight) * self.epoch_pslab
        self.epoch_pslab = self.ema_pslab / (1.0 - self.ema_weight ** (self._epoch + 1.0))

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):

        lb_X = lb_X[0] if isinstance(lb_X, (tuple, list)) else lb_X
        lb_y = lb_y[0] if isinstance(lb_y, (tuple, list)) else lb_y
        ulb_X = ulb_X[0] if isinstance(ulb_X, (tuple, list)) else ulb_X
        lb_logits = self._network(lb_X)
        self.bn_controller.freeze_bn(self._network)
        ulb_logits = self._network(ulb_X)
        self.bn_controller.unfreeze_bn(self._network)
        iter_unlab_pslab = self.epoch_pslab[ulb_idx]
        with torch.no_grad():
            self.epoch_pslab[ulb_idx] = ulb_logits.clone().detach()

        return lb_logits,lb_y,ulb_logits,iter_unlab_pslab

    def optimize(self,loss,*args,**kwargs):
        self._network.zero_grad()
        loss.backward()
        self._optimizer.step()
        if self.ema is not None:
            self.ema.update()

    def get_loss(self,train_result,*args,**kwargs):
        lb_logits,lb_y,ulb_logits,iter_unlab_pslab  = train_result
        sup_loss = Cross_Entropy(reduction='mean')(lb_logits, lb_y)
        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        unsup_loss = _warmup *Consistency(reduction='mean')(ulb_logits,iter_unlab_pslab)
        loss = Semi_Supervised_Loss(self.lambda_u)(sup_loss, unsup_loss)
        return loss

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)



