from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
import numpy as np
from LAMDA_SSL.utils import Bn_Controller
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
import torch.nn as nn
import torch
from LAMDA_SSL.Augmentation.Vision.Mixup import Mixup
import LAMDA_SSL.Config.ICT as config
from LAMDA_SSL.Loss.Semi_Supervised_Loss import Semi_Supervised_Loss

class ICT(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 lambda_u=config.lambda_u,
                 alpha=config.alpha,
                 warmup=config.warmup,
                 mu=config.mu,
                 weight_decay=config.weight_decay,
                 ema_decay=config.ema_decay,
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
                 unlabeled_sampler=config.unlabeled_sampler,
                 labeled_batch_sampler=config.labeled_batch_sampler,
                 unlabeled_batch_sampler=config.unlabeled_batch_sampler,
                 valid_sampler=config.valid_sampler,
                 valid_batch_sampler=config.valid_batch_sampler,
                 test_sampler=config.test_sampler,
                 test_batch_sampler=config.test_batch_sampler,
                 labeled_sampler=config.labeled_sampler,
                 augmentation=config.augmentation,
                 network=config.network,
                 optimizer=config.optimizer,
                 scheduler=config.scheduler,
                 parallel=config.parallel,
                 evaluation=config.evaluation,
                 file=config.file,
                 verbose=config.verbose):
        # >> Parameter:
        # >> - warmup: Warm up ratio for unsupervised loss.
        # >> - lambda_u: The weight of unsupervised loss.
        # >> - alpha: the parameter of Beta distribution in Mixup.
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
        self.ema_decay=ema_decay
        self.lambda_u=lambda_u
        self.weight_decay=weight_decay
        self.warmup=warmup
        self.alpha=alpha
        self.bn_controller=Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def start_fit(self):
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_x = lb_X[0] if isinstance(lb_X, (tuple, list)) else lb_X
        lb_y = lb_y[0] if isinstance(lb_y, (tuple, list)) else lb_y
        ulb_x_1 = ulb_X[0] if isinstance(ulb_X, (tuple, list)) else ulb_X
        lb_logits = self._network(lb_x)
        index = torch.randperm(ulb_x_1.size(0)).to(self.device)
        ulb_x_2=ulb_x_1[index]
        mixup=Mixup(self.alpha)
        if self.ema is not None:
            self.ema.apply_shadow()
        with torch.no_grad():
            ulb_logits_1 = self._network(ulb_x_1)
        if self.ema is not None:
            self.ema.restore()
        ulb_logits_2=ulb_logits_1[index]
        mixed_x= mixup.fit(ulb_x_1).transform(ulb_x_2)
        lam=mixup.lam
        self.bn_controller.freeze_bn(self._network)
        ulb_logits_mix = self._network(mixed_x)
        self.bn_controller.unfreeze_bn(self._network)
        return lb_logits,lb_y,ulb_logits_1,ulb_logits_2,ulb_logits_mix,lam

    def get_loss(self,train_result,*args,**kwargs):
        lb_logits,lb_y,ulb_logits_1,ulb_logits_2,ulb_logits_mix,lam=train_result
        sup_loss = Cross_Entropy(reduction='mean')(lb_logits, lb_y)  # CE_loss for labeled data
        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        unsup_loss = _warmup *Cross_Entropy(use_hard_labels=False, reduction='mean')(ulb_logits_mix,lam * nn.Softmax(dim=-1)(ulb_logits_1)+(1-lam)*
                                                                            nn.Softmax(dim=-1)(ulb_logits_2))

        loss=Semi_Supervised_Loss(lambda_u =self.lambda_u)(sup_loss,unsup_loss)
        return loss

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)



