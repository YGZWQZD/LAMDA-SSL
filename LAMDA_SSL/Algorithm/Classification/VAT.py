from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
import numpy as np
import torch
import LAMDA_SSL.Config.VAT as config
from LAMDA_SSL.utils import _l2_normalize
from torch.autograd import Variable
from LAMDA_SSL.utils import Bn_Controller
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
from LAMDA_SSL.Loss.KL_Divergence import KL_Divergence
from LAMDA_SSL.Loss.EntMin import EntMin

class VAT(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 lambda_u=config.lambda_u,
                 eps=config.eps,
                 warmup=config.warmup,
                 it_vat=config.it_vat,
                 xi=config.xi,
                 lambda_entmin=config.lambda_entmin,
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
                 verbose=config.verbose):
        # >> Parameter
        # >> - lambda_u: The weight of unsupervised loss.
        # >> - num_classes: The number of classes.
        # >> - tsa_schedule: Threshold adjustment strategy, optional 'linear', 'exp' or 'log'.
        # >> - eps: noise level.
        # >> - warmup: The end position of warmup. For example, num_it_total is 100 and warmup is 0.4, then warmup is performed in the first 40 iterations.
        # >> - xi:The scale parameter used when initializing the disturbance variable r, $r=\xi d$. d is a random unit vector.
        # >> - lambda_entmin: Entropy minimizes the weight of the loss.
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
        self.eps=eps
        self.it_vat=it_vat
        self.xi=xi
        self.lambda_entmin=lambda_entmin
        self.bn_controller=Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type


    def start_fit(self):
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y
        ulb_X = ulb_X[0] if isinstance(ulb_X, (tuple, list)) else ulb_X

        lb_logits = self._network(lb_X)

        self.bn_controller.freeze_bn(self._network)
        ulb_logits = self._network(ulb_X)

        d = torch.Tensor(ulb_X.size()).normal_()
        for i in range(self.it_vat):
            d = self.xi*_l2_normalize(d)
            d = Variable(d.to(self.device), requires_grad=True)
            y_hat = self._network(ulb_X + d)
            delta_kl = KL_Divergence(reduction='mean')(ulb_logits.detach(), y_hat)
            delta_kl.backward()
            d = d.grad.data.clone()
            self._network.zero_grad()

        d = _l2_normalize(d)
        d = Variable(d)
        r_adv = self.eps * d
        y_hat = self._network(ulb_X + r_adv.detach())

        self.bn_controller.unfreeze_bn(self._network)
        return lb_logits,lb_y,ulb_logits, y_hat

    def get_loss(self,train_result,*args,**kwargs):
        lb_logits,lb_y,ulb_logits,y_hat=train_result
        _warmup = np.clip(self.it_total / (self.warmup * self.num_it_total),
                a_min=0.0, a_max=1.0)

        sup_loss = Cross_Entropy(reduction='mean')(lb_logits, lb_y)

        unsup_loss = _warmup*KL_Divergence(reduction='mean')(ulb_logits.detach(), y_hat)
        entmin_loss=EntMin(reduction='mean',activation=torch.nn.Softmax(dim=-1))(ulb_logits)

        loss = sup_loss + self.lambda_u * unsup_loss  + self.lambda_entmin * entmin_loss
        return loss

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)


