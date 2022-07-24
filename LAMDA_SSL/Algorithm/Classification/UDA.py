import copy
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
import torch
from LAMDA_SSL.utils import class_status
import math
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
from LAMDA_SSL.Loss.Semi_Supervised_Loss import Semi_Supervised_Loss
import LAMDA_SSL.Config.UDA as config

class UDA(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 threshold=config.threshold,
                 lambda_u=config.lambda_u,
                 T=config.T,
                 num_classes=config.num_classes,
                 tsa_schedule=config.tsa_schedule,
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
        # >> - threshold: The confidence threshold for choosing samples.
        # >> - num_classes: The number of classes.
        # >> - tsa_schedule: Threshold adjustment strategy, optional 'linear', 'exp' or 'log'.
        # >> - T: Sharpening temperature for soft labels.
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
        self.threshold=threshold
        self.tsa_schedule=tsa_schedule
        self.num_classes=num_classes
        self.T=T
        self.weight_decay=weight_decay
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_unlabeled_transform(copy.copy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.strong_augmentation,dim=1,x=1,y=0)

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_classes
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X = lb_X[0] if isinstance(lb_X, (tuple, list)) else lb_X
        lb_y = lb_y[0] if isinstance(lb_y, (tuple, list)) else lb_y
        w_ulb_X,s_ulb_X=ulb_X[0],ulb_X[1]
        num_lb = lb_X.shape[0]
        inputs = torch.cat([lb_X, w_ulb_X, s_ulb_X], dim=0)
        logits = self._network(inputs)
        lb_logits = logits[:num_lb]
        w_ulb_logits, s_ulb_logits = logits[num_lb:].chunk(2)
        return lb_logits,lb_y,w_ulb_logits, s_ulb_logits

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
        lb_logits,lb_y,w_ulb_logits, s_ulb_logits=train_result
        tsa = self.get_tsa()
        sup_mask = torch.max(torch.softmax(lb_logits, dim=-1), dim=-1)[0].le(tsa).float().detach()
        sup_loss = (Cross_Entropy(reduction='none')(lb_logits, lb_y)* sup_mask).mean()  # CE_loss for labeled data
        pseudo_label = torch.softmax(w_ulb_logits, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()
        pseudo_label = torch.softmax(w_ulb_logits / self.T, dim=-1)
        unsup_loss = (Cross_Entropy(reduction='none',use_hard_labels=False)(s_ulb_logits, pseudo_label)*mask ).mean() # MSE loss for unlabeled data
        loss = Semi_Supervised_Loss(self.lambda_u)(sup_loss , unsup_loss)
        return loss

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)


