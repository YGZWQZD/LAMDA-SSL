import copy
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from sklearn.base import ClassifierMixin
from Semi_sklearn.Opitimizer.SemiOptimizer import SemiOptimizer
from Semi_sklearn.Scheduler.SemiScheduler import SemiScheduler
from torch.nn import Softmax
import torch.nn.functional as F
from collections import Counter
from Semi_sklearn.utils import EMA
from Semi_sklearn.utils import class_status
from Semi_sklearn.utils import cross_entropy
import torch

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


class FlexMatch(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
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
                 eval_epoch=None,
                 eval_it=None,
                 optimizer=None,
                 scheduler=None,
                 device='cpu',
                 evaluation=None,
                 threshold=None,
                 lambda_u=None,
                 mu=None,
                 ema_decay=None,
                 T=None,
                 weight_decay=None,
                 num_classes=10,
                 thresh_warmup=None,
                 use_hard_labels=False,
                 use_DA=False,
                 p_target=None
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
                                    evaluation=evaluation
                                    )

        self.lambda_u=lambda_u
        self.threshold=threshold
        self.T=T
        self.num_classes=num_classes
        self.classwise_acc=None
        self.selected_label=None
        self.thresh_warmup=thresh_warmup
        self.use_hard_labels=use_hard_labels
        self.p_model=None
        self.p_target=p_target
        self.use_DA=use_DA
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_unlabeled_transform(copy.deepcopy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.strongly_augmentation,dim=1,x=1,y=0)

    def start_fit(self):
        # print(self._train_dataset.labeled_dataset.y)
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_class
        if self.p_target is None:
            class_counts=torch.Tensor(class_status(self._train_dataset.labeled_dataset.y).class_counts).to(self.device)
            self.p_target = (class_counts / class_counts.sum(dim=-1, keepdim=True))
        self.selected_label = torch.ones((len(self._train_dataset.unlabeled_dataset),), dtype=torch.long, ) * -1
        self.selected_label = self.selected_label.to(self.device)
        self.classwise_acc = torch.zeros((self.num_classes)).to(self.device)
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        w_lb_X=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        w_ulb_X,s_ulb_X=ulb_X[0],ulb_X[1]
        num_lb = w_lb_X.shape[0]
        pseudo_counter = Counter(self.selected_label.tolist())
        if max(pseudo_counter.values()) < len(self._train_dataset.unlabeled_dataset):  # not all(5w) -1
            if self.thresh_warmup:
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
            else:
                wo_negative_one = copy.deepcopy(pseudo_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

        inputs = torch.cat((w_lb_X, w_ulb_X, s_ulb_X))
        logits = self._network(inputs)
        logits_x_lb = logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
        logits_x_ulb_w = logits_x_ulb_w.detach()
        pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)
        if self.use_DA:
            if self.p_model == None:
                self.p_model = torch.mean(pseudo_label.detach(), dim=0)
            else:
                self.p_model = self.p_model * 0.999 + torch.mean(pseudo_label.detach(), dim=0) * 0.001
            pseudo_label = pseudo_label * self.p_target / self.p_model
            pseudo_label = (pseudo_label / pseudo_label.sum(dim=-1, keepdim=True))
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = (max_probs.ge(self.threshold * (self.classwise_acc[max_idx] / (2. - self.classwise_acc[max_idx]))).float()).mean()
        select = max_probs.ge(self.threshold ).long()
        if ulb_idx[select == 1].nelement() != 0:
            self.selected_label[ulb_idx[select == 1]] = max_idx.long()[select == 1]
        if self.use_hard_labels is not True:
            pseudo_label = torch.softmax(logits_x_ulb_w / self.T, dim=-1)
        else:
            pseudo_label=max_idx.long()

        result=(logits_x_lb,lb_y,logits_x_ulb_s,pseudo_label,mask)
        return result

    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb,lb_y,logits_x_ulb_s,pseudo_label,mask = train_result
        Lx = cross_entropy(logits_x_lb, lb_y, reduction='mean')
        if self.use_hard_labels:
            Lu = (cross_entropy(logits_x_ulb_s, pseudo_label, self.use_hard_labels, reduction='none') * mask).mean()
        else:
            Lu = cross_entropy(logits_x_ulb_s, pseudo_label, self.use_hard_labels) * mask.mean()
        loss = Lx + self.lambda_u * Lu
        return loss

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)

