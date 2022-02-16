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
from Semi_sklearn.utils import class_status
from Semi_sklearn.utils import one_hot
from Semi_sklearn.Transform.Mixup import Mixup
import torch.nn.functional as F
from Semi_sklearn.utils import Bn_Controller

# def fix_bn(m,train=False):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         if train:
#             m.train()
#         else:
#             m.eval()

class Mixmatch(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
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
                 lambda_u=None,
                 mu=None,
                 ema_decay=None,
                 weight_decay=None,
                 T=None,
                 num_classes=10,
                 alpha=None
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
        self.weight_decay=weight_decay
        self.warmup=warmup
        self.T=T
        self.alpha=alpha
        self.num_classes=num_classes
        self.bn_controller=Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_unlabled_transform(copy.deepcopy(self.train_dataset.unlabled_transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabled_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabled_transform(self.weakly_augmentation,dim=1,x=1,y=0)

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labled_dataset.y).num_class
        self.it_total = 0
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):

        lb_x=lb_X[0]
        ulb_x_1,ulb_x_2=ulb_X[0],ulb_X[1]

        num_lb = lb_x.shape[0]
        with torch.no_grad():
            # self._network.apply(partial(fix_bn, train=True))
            self.bn_controller.freeze_bn(self._network)
            logits_x_ulb_w1 = self._network(ulb_x_1)
            logits_x_ulb_w2 = self._network(ulb_x_2)
            self.bn_controller.unfreeze_bn(self._network)
            # print(torch.any(torch.isnan(logits_x_ulb_w1)))
            # print(torch.any(torch.isnan(logits_x_ulb_w2)))

            avg_prob_x_ulb = (torch.softmax(logits_x_ulb_w1, dim=1) + torch.softmax(logits_x_ulb_w2, dim=1)) / 2
            avg_prob_x_ulb = (avg_prob_x_ulb / avg_prob_x_ulb.sum(dim=-1, keepdim=True))
            # sharpening
            sharpen_prob_x_ulb = avg_prob_x_ulb ** (1 / self.T)
            sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()
            input_labels = torch.cat(
                [one_hot(lb_y, self.num_classes).to(self.device), sharpen_prob_x_ulb, sharpen_prob_x_ulb], dim=0)
            inputs = torch.cat([lb_x, ulb_x_1, ulb_x_2])
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
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        return logits_x,mixed_y[:num_lb],logits_u,mixed_y[num_lb:]

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
        logits_x_lb,lb_y,logits_x_ulb,ulb_y=train_result
        sup_loss = cross_entropy(logits_x_lb, lb_y,use_hard_labels=False).mean()  # CE_loss for labeled data
        unsup_loss=F.mse_loss(torch.softmax(logits_x_ulb, dim=-1), ulb_y, reduction='mean')
        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        loss = sup_loss + self.lambda_u * _warmup * unsup_loss
        return loss

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)



