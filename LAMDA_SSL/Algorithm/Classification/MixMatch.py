import copy
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
import torch
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
from LAMDA_SSL.Loss.Consistency import Consistency
from LAMDA_SSL.Loss.Semi_Supervised_Loss import Semi_Supervised_Loss
import numpy as np
from LAMDA_SSL.utils import class_status
from LAMDA_SSL.utils import one_hot
from LAMDA_SSL.Augmentation.Vision.Mixup import Mixup
from LAMDA_SSL.utils import Bn_Controller
import LAMDA_SSL.Config.MixMatch as config

class MixMatch(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 lambda_u=config.lambda_u,
                 T=config.T,
                 num_classes=config.num_classes,
                 alpha=config.alpha,
                 warmup=config.warmup,
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
        # >> Parameter:
        # >> - lambda_u: The weight of unsupervised loss.
        # >> - T: Sharpening temperature for soft labels.
        # >> - num_classes: The number of classes.
        # >> - alpha: The parameter of the beta distribution in Mixup.
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
        self.T=T
        self.alpha=alpha
        self.num_classes=num_classes
        self.bn_controller=Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_unlabeled_transform(copy.copy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=1,y=0)

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_classes
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):

        lb_X = lb_X[0] if isinstance(lb_X, (tuple, list)) else lb_X
        lb_y = lb_y[0] if isinstance(lb_y, (tuple, list)) else lb_y
        ulb_x_1,ulb_x_2=ulb_X[0],ulb_X[1]

        num_lb = lb_X.shape[0]
        with torch.no_grad():
            self.bn_controller.freeze_bn(self._network)
            ulb_logits_1 = self._network(ulb_x_1)
            ulb_logits_2 = self._network(ulb_x_2)
            self.bn_controller.unfreeze_bn(self._network)

            ulb_avg_prob_X = (torch.softmax(ulb_logits_1, dim=1) + torch.softmax(ulb_logits_2, dim=1)) / 2
            ulb_avg_prob_X = (ulb_avg_prob_X / ulb_avg_prob_X.sum(dim=-1, keepdim=True))
            # sharpening
            ulb_sharpen_prob_X = ulb_avg_prob_X ** (1 / self.T)
            ulb_sharpen_prob_X = (ulb_sharpen_prob_X / ulb_sharpen_prob_X.sum(dim=-1, keepdim=True)).detach()
            input_labels = torch.cat(
                [one_hot(lb_y, self.num_classes,device=self.device).to(self.device), ulb_sharpen_prob_X, ulb_sharpen_prob_X], dim=0)
            inputs = torch.cat([lb_X, ulb_x_1, ulb_x_2])
            index = torch.randperm(inputs.size(0)).to(self.device)

            mixed_x, mixed_y=Mixup(self.alpha).fit((inputs,input_labels)).transform((inputs[index],input_labels[index]))
            mixed_x = list(torch.split(mixed_x, num_lb))
            mixed_x = self.interleave(mixed_x, num_lb)

        _mix_0=self._network(mixed_x[0])
        logits = [_mix_0]
        # calculate BN for only the first batch
        self.bn_controller.freeze_bn(self._network)
        for ipt in mixed_x[1:]:
            _mix_i=self._network(ipt)
            logits.append(_mix_i)

        # put interleaved samples back
        logits = self.interleave(logits, num_lb)
        self.bn_controller.unfreeze_bn(self._network)
        lb_logits = logits[0]
        ulb_logits = torch.cat(logits[1:], dim=0)
        return lb_logits,mixed_y[:num_lb],ulb_logits,mixed_y[num_lb:]

    def interleave_offsets(self, batch, num):
        groups = [batch // (num + 1)] * (num + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        num = len(xy) - 1
        offsets = self.interleave_offsets(batch, num)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(num + 1)] for v in xy]
        for i in range(1, num + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    def get_loss(self,train_result,*args,**kwargs):
        lb_logits,lb_y,ulb_logits,ulb_y=train_result
        sup_loss = Cross_Entropy(use_hard_labels=False,reduction='mean')(lb_logits, lb_y)
        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        unsup_loss=_warmup * Consistency(reduction='mean')(torch.softmax(ulb_logits, dim=-1), ulb_y)
        loss = Semi_Supervised_Loss(self.lambda_u )(sup_loss ,  unsup_loss)
        return loss

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)



