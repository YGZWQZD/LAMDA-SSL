import copy
from lamda_ssl.Base.InductiveEstimator import InductiveEstimator
from lamda_ssl.Base.SemiDeepModelMixin import SemiDeepModelMixin
from sklearn.base import ClassifierMixin
import lamda_ssl.Config.Fixmatch as config
import torch.nn.functional as F

import torch



class Fixmatch(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
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
                 parallel=None,
                 epoch=1,
                 num_it_epoch=None,
                 num_it_total=None,
                 eval_epoch=None,
                 eval_it=None,
                 optimizer=None,
                 scheduler=None,
                 device='cpu',
                 evaluation=None,
                 file=None,
                 threshold=0.95,
                 lambda_u=1.0,
                 mu=1.0,
                 ema_decay=0.999,
                 T=0.5,
                 weight_decay=5e-4
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
        self.lambda_u=lambda_u
        self.threshold=threshold
        self.T=T
        self.weight_decay=weight_decay
        self._estimator_type=ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_unlabeled_transform(copy.deepcopy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.strongly_augmentation,dim=1,x=1,y=0)

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        w_lb_X=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y
        w_ulb_X,s_ulb_X=ulb_X[0],ulb_X[1]
        batch_size = w_lb_X.shape[0]
        inputs=torch.cat((w_lb_X, w_ulb_X, s_ulb_X))
        logits = self._network(inputs)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        result=(logits_x,lb_y,logits_u_w,logits_u_s)
        return result

    def get_loss(self,train_result,*args,**kwargs):
        logits_x, lb_y, logits_u_w, logits_u_s = train_result
        # Lx = F.cross_entropy(logits_x, lb_y, reduction='mean')
        log_pred = F.log_softmax(logits_x, dim=-1)
        Lx=F.nll_loss(log_pred, lb_y, reduction='mean')
        pseudo_label = torch.softmax(logits_u_w.detach() / self.T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()

        Lu = (F.cross_entropy(logits_u_s, targets_u,
                              reduction='none') * mask).mean()

        loss = Lx + self.lambda_u * Lu
        return loss

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)