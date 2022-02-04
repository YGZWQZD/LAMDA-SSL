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
from Semi_sklearn.Data_Augmentation.Mixup import Mixup
import torch.nn.functional as F

def fix_bn(m,train=False):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if train:
            m.train()
        else:
            m.eval()

class Mixmatch(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
    def __init__(self,train_dataset=None,test_dataset=None,
                 train_dataloader=None,
                 test_dataloader=None,
                 augmentation=None,
                 network=None,
                 train_sampler=None,
                 train_batch_sampler=None,
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
                                    test_dataset=test_dataset,
                                    train_dataloader=train_dataloader,
                                    test_dataloader=test_dataloader,
                                    augmentation=augmentation,
                                    network=network,
                                    train_sampler=train_sampler,
                                    train_batch_sampler=train_batch_sampler,
                                    test_sampler=test_sampler,
                                    test_batch_Sampler=test_batch_sampler,
                                    epoch=epoch,
                                    num_it_epoch=num_it_epoch,
                                    num_it_total=num_it_total,
                                    eval_epoch=eval_epoch,
                                    eval_it=eval_it,
                                    mu=mu,
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

        if self.ema_decay is not None:
            self.ema=EMA(model=self._network,decay=ema_decay)
            self.ema.register()
        else:
            self.ema=None
        if isinstance(self._augmentation,dict):
            self.weakly_augmentation=self._augmentation['augmentation']
            self.normalization = self._augmentation['normalization']
        elif isinstance(self._augmentation,(list,tuple)):
            self.weakly_augmentation = self._augmentation[0]
            self.normalization = self._augmentation[1]
        else:
            self.weakly_augmentation = copy.deepcopy(self._augmentation)
            self.normalization = copy.deepcopy(self._augmentation)

        if isinstance(self._optimizer,SemiOptimizer):
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in self._network.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self._network.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self._optimizer=self._optimizer.init_optimizer(params=grouped_parameters)

        if isinstance(self._scheduler,SemiScheduler):
            self._scheduler=self._scheduler.init_scheduler(optimizer=self._optimizer)

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labled_dataset.y).num_class

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):

        lb_x = self.weakly_augmentation.fit_transform(copy.deepcopy(lb_X))
        ulb_x_1 = self.weakly_augmentation.fit_transform(copy.deepcopy(ulb_X))
        ulb_x_2 = self.weakly_augmentation.fit_transform(copy.deepcopy(ulb_X))

        num_lb = lb_x.shape[0]
        with torch.no_grad():
            self._network.apply(partial(fix_bn,train=False))
            logits_x_ulb_w1 = self._network(ulb_x_1)
            logits_x_ulb_w2 = self._network(ulb_x_2)
            self._network.apply(partial(fix_bn, train=True))
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
        logits = [self._network(mixed_x[0])]
        # calculate BN for only the first batch
        self._network.apply(partial(fix_bn,train=False))
        for ipt in mixed_x[1:]:
            logits.append(self._network(ipt))

        # put interleaved samples back
        logits = self.interleave(logits, num_lb)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        self._network.apply(partial(fix_bn,train=True))

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

    def get_predict_result(self,y_est,*args,**kwargs):

        self.y_score=Softmax(dim=-1)(y_est)
        max_probs,y_pred=torch.max(self.y_score, dim=-1)
        return y_pred

    def predict(self,X=None):
        return SemiDeepModelMixin.predict(self,X=X)

    def optimize(self,*args,**kwargs):
        self._optimizer.step()
        self._scheduler.step()
        if self.ema is not None:
            self.ema.update()
        self._network.zero_grad()

    def estimate(self,X,idx=None,*args,**kwargs):
        X=self.normalization.fit_transform(X)
        if self.ema is not None:
            self.ema.apply_shadow()
        outputs = self._network(X)
        if self.ema is not None:
            self.ema.restore()
        return outputs



