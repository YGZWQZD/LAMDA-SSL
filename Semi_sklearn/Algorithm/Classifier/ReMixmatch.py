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



class ReMixmatch(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
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
                 warmup=None,
                 eval_epoch=None,
                 eval_it=None,
                 optimizer=None,
                 scheduler=None,
                 device='cpu',
                 evaluation=None,
                 parallel=None,
                 file=None,
                 lambda_u=None,
                 mu=None,
                 ema_decay=None,
                 weight_decay=None,
                 T=None,
                 num_classes=10,
                 alpha=None,
                 p_target=None,
                 lambda_s=None,
                 lambda_rot=None,
                 rotate_v_list=None
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
        self.lambda_s=lambda_s
        self.lambda_rot=lambda_rot
        self.weight_decay=weight_decay
        self.warmup=warmup
        self.T=T
        self.alpha=alpha
        self.num_classes=num_classes
        self.rotate_v_list=rotate_v_list if rotate_v_list is not None else [0, 90, 180, 270]
        self.p_model = None
        self.p_target=p_target
        self.bn_controller = Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_unlabeled_transform(copy.deepcopy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_unlabeled_transform(copy.deepcopy(self.train_dataset.unlabeled_transform), dim=0, x=2)
        self._train_dataset.add_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.strongly_augmentation,dim=1,x=1,y=0)
        self._train_dataset.add_unlabeled_transform(self.strongly_augmentation, dim=1, x=2, y=0)

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_class
        if self.p_target is None:
            class_counts=torch.Tensor(class_status(self._train_dataset.labeled_dataset.y).class_counts).to(self.device)
            self.p_target = (class_counts / class_counts.sum(dim=-1, keepdim=True))
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_x_w=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y

        ulb_x_w, ulb_x_s_1, ulb_x_s_2 = ulb_X[0],ulb_X[1],ulb_X[2]

        ulb_x_s_1_rot = torch.Tensor().to(self.device)
        rot_index = []
        for item in ulb_x_s_1:
            _v = np.random.choice(self.rotate_v_list, 1).item()
            ulb_x_s_1_rot = torch.cat((ulb_x_s_1_rot, Rotate(v=_v).fit_transform(item).unsqueeze(0)), dim=0)
            rot_index.append(self.rotate_v_list.index(_v))

        rot_index = torch.LongTensor(rot_index).to(self.device)
        num_lb = lb_x_w.shape[0]

        with torch.no_grad():
            # self._network.apply(partial(fix_bn, train=True))
            # self._network.apply(partial(fix_bn,train=False))
            self.bn_controller.freeze_bn(model=self._network)
            logits_x_ulb_w = self._network(ulb_x_w)[0]
            self.bn_controller.unfreeze_bn(model=self._network)
            # print(torch.any(torch.isnan(logits_x_ulb_w)))
            prob_x_ulb = torch.softmax(logits_x_ulb_w, dim=1)
            if self.p_model is None:
                self.p_model = torch.mean(prob_x_ulb.detach(), dim=0).to(self.device)
            else:
                self.p_model = self.p_model * 0.999 + torch.mean(prob_x_ulb.detach(), dim=0) * 0.001
            prob_x_ulb = prob_x_ulb * self.p_target / self.p_model
            prob_x_ulb = (prob_x_ulb / prob_x_ulb.sum(dim=-1, keepdim=True))
            sharpen_prob_x_ulb = prob_x_ulb ** (1 / self.T)
            sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()
            mixed_inputs = torch.cat((lb_x_w, ulb_x_s_1, ulb_x_s_2, ulb_x_w))
            input_labels = torch.cat(
                [one_hot(lb_y, self.num_classes,self.device).to(self.device), sharpen_prob_x_ulb, sharpen_prob_x_ulb,
                 sharpen_prob_x_ulb], dim=0)
            index = torch.randperm(mixed_inputs.size(0)).to(self.device)

            mixed_x, mixed_y = Mixup(self.alpha).fit((mixed_inputs, input_labels)).transform(
                (mixed_inputs[index], input_labels[index]))
            mixed_x = list(torch.split(mixed_x, num_lb))
            mixed_x = self.interleave(mixed_x, num_lb)

        _mix_0 = self._network(mixed_x[0])[0]
        logits = [_mix_0]
        self.bn_controller.freeze_bn(model=self._network)
        for ipt in mixed_x[1:]:
            _mix_i = self._network(ipt)[0]
            logits.append(_mix_i)

        u1_logits = self._network(ulb_x_s_1)[0]
        logits_rot = self._network(ulb_x_s_1_rot)[1]
        logits = self.interleave(logits, num_lb)
        self.bn_controller.unfreeze_bn(model=self._network)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:])
        return logits_x,mixed_y[:num_lb],logits_u,mixed_y[num_lb:],u1_logits,sharpen_prob_x_ulb,logits_rot,rot_index

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


    def get_loss(self,train_result,*args,**kwargs):
        mix_lb_x,mix_lb_y,mix_ulb_x,mix_ulb_y,s_x,s_y,rot_x,rot_y=train_result
        sup_loss = cross_entropy(mix_lb_x, mix_lb_y,use_hard_labels=False).mean()  # CE_loss for labeled data
        unsup_loss=cross_entropy(mix_ulb_x, mix_ulb_y,use_hard_labels=False).mean()
        s_loss=cross_entropy(s_x, s_y,use_hard_labels=False).mean()
        # print(rot_y.dtype)
        rot_loss = cross_entropy(rot_x, rot_y, reduction='mean').mean()
        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        loss = sup_loss + self.lambda_u * _warmup * unsup_loss+self.lambda_s * _warmup *s_loss+self.lambda_rot*rot_loss
        return loss

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)



