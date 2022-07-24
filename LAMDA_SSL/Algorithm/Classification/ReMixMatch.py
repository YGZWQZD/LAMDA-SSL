import copy
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
import torch
import numpy as np
from LAMDA_SSL.utils import class_status
from LAMDA_SSL.utils import one_hot
from LAMDA_SSL.Augmentation.Vision.Mixup import Mixup
from LAMDA_SSL.Augmentation.Vision.Rotate import Rotate
from LAMDA_SSL.utils import Bn_Controller
import LAMDA_SSL.Config.ReMixMatch as config
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy


class ReMixMatch(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 alpha=config.alpha,
                 lambda_u=config.lambda_u,
                 T=config.T,
                 num_classes=config.num_classes,
                 warmup=config.warmup,
                 p_target=config.p_target,
                 lambda_s=config.lambda_s,
                 lambda_rot=config.lambda_rot,
                 rotate_v_list=config.rotate_v_list,
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
        # >> - p_target: The target distribution of labeled data.
        # >> - lambda_s: The weight for unsupervised loss computed based on pre-mixup data.
        # >> - lambda_rot: The weight of rotation angle classification loss.
        # >> - rotate_v_list: A list of rotation angles.
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
        self.lambda_s=lambda_s
        self.lambda_rot=lambda_rot
        self.weight_decay=weight_decay
        self.warmup=warmup
        self.T=T
        self.alpha=alpha
        self.num_classes=num_classes
        self.rotate_v_list=rotate_v_list
        self.p_model = None
        self.p_target=p_target
        self.bn_controller = Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_unlabeled_transform(copy.copy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_unlabeled_transform(copy.copy(self.train_dataset.unlabeled_transform), dim=0, x=2)
        self._train_dataset.add_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.strong_augmentation,dim=1,x=1,y=0)
        self._train_dataset.add_unlabeled_transform(self.strong_augmentation, dim=1, x=2, y=0)

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_classes
        if self.p_target is None:
            class_counts=torch.Tensor(class_status(self._train_dataset.labeled_dataset.y).class_counts).to(self.device)
            self.p_target = (class_counts / class_counts.sum(dim=-1, keepdim=True))
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y

        w_ulb_X, s_ulb_X_1, s_ulb_X_2 = ulb_X[0],ulb_X[1],ulb_X[2]

        s_ulb_rot_X_1 = torch.Tensor().to(self.device)
        rot_index = []
        for item in s_ulb_X_1:
            _v = np.random.choice(self.rotate_v_list, 1).item()
            s_ulb_rot_X_1 = torch.cat((s_ulb_rot_X_1, Rotate(v=_v).fit_transform(item).unsqueeze(0)), dim=0)
            rot_index.append(self.rotate_v_list.index(_v))

        rot_index = torch.LongTensor(rot_index).to(self.device)
        num_lb = lb_X.shape[0]

        with torch.no_grad():
            self.bn_controller.freeze_bn(model=self._network)
            w_ulb_logits = self._network(w_ulb_X)[0]
            self.bn_controller.unfreeze_bn(model=self._network)
            ulb_prob = torch.softmax(w_ulb_logits, dim=1)
            if self.p_model is None:
                self.p_model = torch.mean(ulb_prob.detach(), dim=0).to(self.device)
            else:
                self.p_model = self.p_model * 0.999 + torch.mean(ulb_prob.detach(), dim=0) * 0.001
            ulb_prob = ulb_prob * self.p_target / self.p_model
            ulb_prob = (ulb_prob / ulb_prob.sum(dim=-1, keepdim=True))
            ulb_sharpen_prob = ulb_prob ** (1 / self.T)
            ulb_sharpen_prob = (ulb_sharpen_prob / ulb_sharpen_prob.sum(dim=-1, keepdim=True)).detach()
            mixed_inputs = torch.cat((lb_X, s_ulb_X_1, s_ulb_X_2, w_ulb_X))
            input_labels = torch.cat(
                [one_hot(lb_y, self.num_classes,self.device).to(self.device), ulb_sharpen_prob, ulb_sharpen_prob,
                 ulb_sharpen_prob], dim=0)
            index = torch.randperm(mixed_inputs.size(0)).to(self.device)

            mixed_X, mixed_y = Mixup(self.alpha).fit((mixed_inputs, input_labels)).transform(
                (mixed_inputs[index], input_labels[index]))
            mixed_X = list(torch.split(mixed_X, num_lb))
            mixed_X = self.interleave(mixed_X, num_lb)

        _mix_0 = self._network(mixed_X[0])[0]
        logits = [_mix_0]
        self.bn_controller.freeze_bn(model=self._network)
        for ipt in mixed_X[1:]:
            _mix_i = self._network(ipt)[0]
            logits.append(_mix_i)

        s_ulb_logits_1 = self._network(s_ulb_X_1)[0]
        rot_logits = self._network(s_ulb_rot_X_1)[1]
        logits = self.interleave(logits, num_lb)
        self.bn_controller.unfreeze_bn(model=self._network)
        lb_logits = logits[0]
        ulb_logits = torch.cat(logits[1:])
        return lb_logits,mixed_y[:num_lb],ulb_logits,mixed_y[num_lb:],s_ulb_logits_1,ulb_sharpen_prob,rot_logits,rot_index

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
        lb_logits,mix_lb_y,ulb_logits,mix_ulb_y,s_ulb_logits_1,ulb_sharpen_prob,rot_logits,rot_index=train_result
        sup_loss = Cross_Entropy(use_hard_labels=False,reduction='mean')(lb_logits, mix_lb_y)  # CE_loss for labeled data
        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        unsup_loss=_warmup *Cross_Entropy(use_hard_labels=False,reduction='mean')(ulb_logits, mix_ulb_y)
        s_loss=_warmup *Cross_Entropy(use_hard_labels=False,reduction='mean')(s_ulb_logits_1,ulb_sharpen_prob)
        rot_loss = Cross_Entropy(reduction='mean')(rot_logits,rot_index)
        loss = sup_loss + self.lambda_u *  unsup_loss+self.lambda_s * s_loss+self.lambda_rot*rot_loss
        return loss

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)



