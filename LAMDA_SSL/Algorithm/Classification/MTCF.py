import copy
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
import LAMDA_SSL.Config.FixMatch as config
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
from LAMDA_SSL.Loss.Semi_Supervised_Loss import Semi_Supervised_Loss
from LAMDA_SSL.utils import class_status
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
import torch

class MTCF(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
    			 alpha=0.75,
                 lambda_u=75,
                 T=0.5,
                 num_classes=None,
                 mu=config.mu,
                 weight_decay=config.weight_decay,
                 ema_decay=config.ema_decay,
                 epoch=config.epoch,
                 num_it_epoch=config.num_it_epoch,
                 num_it_total=config.num_it_total,
                 eval_epoch=config.eval_epoch,
                 eval_it=config.eval_it,
                 optimizer=config.optimizer,
                 scheduler=config.scheduler,
                 device=config.device,
                 train_dataset=config.train_dataset,
                 labeled_dataset=config.labeled_dataset,
                 unlabeled_dataset=config.unlabeled_dataset,
                 valid_dataset=config.valid_dataset,
                 test_dataset=config.test_dataset,
                 train_dataloader=config.train_dataloader,
                 valid_dataloader=config.valid_dataloader,
                 test_dataloader=config.test_dataloader,
                 augmentation=config.augmentation,
                 network=config.network,
                 train_sampler=config.train_sampler,
                 train_batch_sampler=config.train_batch_sampler,
                 valid_sampler=config.valid_sampler,
                 valid_batch_sampler=config.valid_batch_sampler,
                 test_sampler=config.test_sampler,
                 test_batch_sampler=config.test_batch_sampler,
                 labeled_dataloader=config.labeled_dataloader,
                 unlabeled_dataloader=config.unlabeled_dataloader,
                 labeled_sampler=config.labeled_sampler,
                 unlabeled_sampler=config.unlabeled_sampler,
                 labeled_batch_sampler=config.labeled_batch_sampler,
                 unlabeled_batch_sampler=config.unlabeled_batch_sampler,
                 parallel=config.parallel,
                 evaluation=config.evaluation,
                 file=config.file,
                 verbose=config.verbose
                 ):

        # >> Parameter:
        # >> - threshold: The confidence threshold for choosing samples.
        # >> - lambda_u: The weight of unsupervised loss.
        # >> - T: Sharpening temperature.
        # >> - num_classes: The number of classes for the classification task.
        # >> - thresh_warmup: Whether to use threshold warm-up mechanism.
        # >> - use_hard_labels: Whether to use hard labels in the consistency regularization.
        # >> - use_DA: Whether to perform distribution alignment for soft labels.
        # >> - p_target: p(y) based on the labeled examples seen during training.

        DeepModelMixin.__init__(self,train_dataset=train_dataset,
                                    valid_dataset=valid_dataset,
                                    labeled_dataset=labeled_dataset,
                                    unlabeled_dataset=unlabeled_dataset,
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
        self.lambda_u=lambda_u
        self.alpha=alpha
        self.T=T
        self.weight_decay=weight_decay
        self.num_classes=num_classes
        self._estimator_type=ClassifierMixin._estimator_type

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_classes
        self._network.zero_grad()
        self._network.train()

    def init_transform(self):
        self._train_dataset.add_unlabeled_transform(copy.copy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.strong_augmentation,dim=1,x=1,y=0)

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        # assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y
        inputs_u,inputs_u2=ulb_X[0],ulb_X[1]
        batch_size = lb_X.shape[0]
        targets_x = torch.zeros(batch_size, self.num_classes).to(self.device).scatter_(1, lb_y.view(-1,1), 1)
        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = self._network(inputs_u)
            outputs_u2 = self._network(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/self.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([lb_X, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(self.alpha, self.alpha)

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = self.interleave(mixed_input, batch_size)

        logits = [self._network(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(self._network(input))

        # put interleaved samples back
        logits = self.interleave(logits, batch_size)

        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        train_result=(logits_x,mixed_target[:batch_size],logits_u,mixed_target[batch_size:])
        return train_result



    def get_loss(self,train_result,*args,**kwargs):
        lb_logits, lb_y, w_ulb_logits, s_ulb_logits = train_result
        w = self._epoch+self.it_epoch/self.num_it_epoch
        Lx = nn.CrossEntropyLoss()(lb_logits, lb_y)
        Lu = nn.CrossEntropyLoss()(w_ulb_logits, s_ulb_logits)
        loss = Lx + w*Lu
        return loss

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)