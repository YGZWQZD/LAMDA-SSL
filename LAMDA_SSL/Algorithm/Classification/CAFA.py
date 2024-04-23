from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
from LAMDA_SSL.Loss.Consistency import Consistency
from LAMDA_SSL.Loss.Semi_Supervised_Loss import Semi_Supervised_Loss
from LAMDA_SSL.utils import Bn_Controller
from LAMDA_SSL.Network.AdversarialNet import AdversarialNet
import copy
import numpy as np
import LAMDA_SSL.Config.CAFA as config
import torch
import torch.nn.functional as F
import torch.nn as nn
from LAMDA_SSL.Base.BaseOptimizer import BaseOptimizer
from LAMDA_SSL.Base.BaseScheduler import BaseScheduler
from LAMDA_SSL.utils import class_status
import math
def TempScale(p, t):
    return p / t

def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))

def compute_score(inputs, model, eps):
    model.eval()
    inputs.requires_grad = True
    _, output = model(inputs)
    softmax_output = output.softmax(1)
    softmax_output = TempScale(softmax_output, 0.5)
    max_value, max_target = torch.max(softmax_output, dim=1)
    xent = F.cross_entropy(softmax_output, max_target.detach().long())
    d = torch.autograd.grad(xent, inputs)[0]
    d = torch.ge(d, 0)
    d = (d.float() - 0.5) * 2
    # Normalizing the gradient to the same space of image
    # d[0][0] = (d[0][0]) / (63.0 / 255.0)
    # d[0][1] = (d[0][1]) / (62.1 / 255.0)
    # d[0][2] = (d[0][2]) / (66.7 / 255.0)
    inputs_hat = torch.add(inputs.data, -eps, d.detach())
    _, output_hat = model(inputs_hat)
    softmax_output_hat = output_hat.softmax(1)
    softmax_output_hat = TempScale(softmax_output_hat, 0.5)
    max_value_hat = torch.max(softmax_output_hat, dim=1).values
    pred_shift = torch.abs(max_value - max_value_hat).unsqueeze(1)
    model.train()
    return pred_shift.detach()

def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    x = x / max(torch.mean(x), 1e-6)
    return x.detach()

def feature_scaling(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x.detach()

def pseudo_label_calibration(pslab, weight):
    weight = weight.transpose(1, 0).expand(pslab.shape[0], -1)
    weight = normalize_weight(weight)
    pslab = torch.exp(pslab)
    pslab = pslab * weight
    pslab = pslab / torch.sum(pslab, 1, keepdim=True)
    return pslab

def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)

def get_label_share_weight(domain_out, pred_shift, domain_temperature=1.0, class_temperature=1.0):
    min_val = pred_shift.min()
    max_val = pred_shift.max()
    pred_shift = (pred_shift - min_val) / (max_val - min_val)
    pred_shift = reverse_sigmoid(pred_shift)
    pred_shift = pred_shift / class_temperature
    pred_shift = nn.Sigmoid()(pred_shift)

    domain_logit = reverse_sigmoid(domain_out)
    domain_logit = domain_logit / domain_temperature
    domain_out = nn.Sigmoid()(domain_logit)

    weight = domain_out - pred_shift
    weight = weight.detach()
    return weight


def get_unlabel_share_weight(domain_out, pred_shift, domain_temperature=1.0, class_temperature=1.0):
    weight = get_label_share_weight(domain_out, pred_shift, domain_temperature, class_temperature)
    return -weight


def match_string(stra, strb):
    '''
        stra: labels.
        strb: unlabeled data predicts.
    '''
    l_b, prob = torch.argmax(strb, dim=1), torch.max(strb, dim=1).values
    permidx = torch.tensor(range(len(l_b)))

    for i in range(len(l_b)):
        if stra[i] != l_b[i]:
            mask = (l_b[i:] == stra[i]).float()
            if mask.sum() > 0:
                idx_tmp = int(i + torch.argmax(prob[i:] * mask, dim=0))
                tmp = permidx[i].data.clone()
                permidx[i] = permidx[idx_tmp]
                permidx[idx_tmp] = tmp
    return permidx

def compute_class_weight(weight, label, class_weight):
        for i in range(len(class_weight)):
            mask = (label == i)
            class_weight[i] = weight[mask].mean()
        return class_weight

class CAFA(DeepModelMixin,InductiveEstimator,ClassifierMixin):
    def __init__(self,lambda_u=config.lambda_u,
                 warmup=config.warmup,
                 mu=config.mu,
                 threshold=config.threshold,
                 T=config.T,
                 ema_decay=config.ema_decay,
                 adv_warmup=config.adv_warmup,
                 weight_decay=config.weight_decay,
                 eps=config.eps,
                 l_domain_temper=config.l_domain_temper,
                 u_domain_temper=config.u_domain_temper,
                 l_class_temper=config.l_class_temper,
                 u_class_temper=config.u_class_temper,
                 num_classes=config.num_classes,
                 discriminator=config.discriminator,
                 discriminator_separate=config.discriminator_separate,
                 discriminator_optimizer=config.discriminator_optimizer,
                 discriminator_optimizer_separate=config.discriminator_optimizer_separate,
                 discriminator_scheduler=config.discriminator_scheduler,
                 discriminator_scheduler_separate=config.discriminator_scheduler_separate,
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
        # >> - warmup: The end position of warmup. For example, num_it_total is 100 and warmup is 0.4,
        #              then warmup is performed in the first 40 iterations.
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
        self.adv_warmup=adv_warmup
        self.threshold=threshold
        self.T=T
        self.eps=eps
        self.num_classes=num_classes
        self.discriminator = discriminator
        self.discriminator_separate = discriminator_separate
        self.discriminator_optimizer=discriminator_optimizer
        self.discriminator_optimizer_separate=discriminator_optimizer_separate
        self.discriminator_scheduler=discriminator_scheduler
        self.discriminator_scheduler_separate=discriminator_scheduler_separate
        self.l_domain_temper=l_domain_temper
        self.u_domain_temper=u_domain_temper
        self.l_class_temper=l_class_temper
        self.u_class_temper=u_class_temper
        self._discriminator=copy.deepcopy(discriminator)
        self._discriminator_separate = copy.deepcopy(discriminator_separate)
        self._discriminator_optimizer=copy.deepcopy(discriminator_optimizer)
        self._discriminator_optimizer_separate = copy.deepcopy(discriminator_optimizer_separate)
        self._discriminator_scheduler=copy.deepcopy(discriminator_scheduler)
        self._discriminator_scheduler_separate = copy.deepcopy(discriminator_scheduler_separate)
        self.bn_controller=Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_unlabeled_transform(copy.copy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.strong_augmentation,dim=1,x=1,y=0)

    def init_model(self):
        self._network = copy.deepcopy(self.network)
        self._parallel = copy.deepcopy(self.parallel)
        self._discriminator=copy.deepcopy(self.discriminator)
        self._discriminator_separate = copy.deepcopy(self.discriminator_separate)
        if self.device is None:
            self.device='cpu'
        if self.device is not 'cpu':
            torch.cuda.set_device(self.device)
        self._discriminator=self._discriminator.to(self.device)
        self._discriminator_separate=self._discriminator_separate.to(self.device)
        self._network=self._network.to(self.device)
        if self._parallel is not None:
            self._network=self._parallel.init_parallel(self._network)
            self._discriminator=self._parallel.init_parallel(self._discriminator)
            self._discriminator_separate=self._parallel.init_parallel(self._discriminator_separate)

    def start_fit(self, *args, **kwargs):
        self.init_epoch()
        self._network.zero_grad()
        self._network.train()
        self._discriminator.zero_grad()
        self._discriminator.train()
        self._discriminator_separate.zero_grad()
        self._discriminator_separate.train()
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_classes
        self.l_weight = torch.zeros((len(self._train_dataset.labeled_dataset), 1)).to(self.device)
        self.u_weight = torch.zeros((len(self._train_dataset.unlabeled_dataset), 1)).to(self.device)
        self.class_weight = torch.zeros((self.num_classes, 1)).to(self.device)
        self.label_all = torch.zeros(len(self._train_dataset.labeled_dataset)).to(self.device).long()
        self.beta_distribution = torch.distributions.beta.Beta(0.75, 0.75)

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y
        w_ulb_X,s_ulb_X=ulb_X[0],ulb_X[1]
        inputs=torch.cat((lb_X, w_ulb_X, s_ulb_X))
        batch_size=lb_X.shape[0]
        features,logits = self._network(inputs)
        lb_logits = logits[:batch_size]
        w_ulb_logits, s_ulb_logits = logits[batch_size:].chunk(2)
        l_feature=features[:batch_size]
        u_feature,_=features[batch_size:].chunk(2)

        self._network.eval()
        l_pred_shift = compute_score(lb_X.detach(), self._network,self.eps).detach()
        u_pred_shift = compute_score(ulb_X[0].detach(),self._network,self.eps).detach()
        self._network.train()

        l_domain_prob = self._discriminator.forward(l_feature)
        u_domain_prob = self._discriminator.forward(u_feature)

        permidx = match_string(lb_y, w_ulb_logits)

        shuf_u_feature = u_feature[permidx]
        cos_sim = nn.CosineSimilarity(dim=1)(l_feature, shuf_u_feature)
        cos_sim = feature_scaling(cos_sim)
        cos_sim = cos_sim.unsqueeze(1).detach()
        lam = self.beta_distribution.sample().item()
        lam = max(lam, 1 - lam)

        mix_feature = lam * l_feature + (1 - lam) * shuf_u_feature

        domain_prob_separate_mix = self._discriminator_separate(mix_feature.detach())
        l_domain_prob_separate = self._discriminator_separate.forward(l_feature.detach())
        u_domain_prob_separate = self._discriminator_separate.forward(u_feature.detach())

        label_share_weight = get_label_share_weight(
            l_domain_prob_separate, l_pred_shift, domain_temperature=self.l_domain_temper,
            class_temperature=self.l_class_temper)
        label_share_weight = normalize_weight(label_share_weight)

        unlabel_share_weight = get_unlabel_share_weight(
            u_domain_prob_separate, u_pred_shift, domain_temperature=self.u_domain_temper,
            class_temperature=self.u_class_temper)
        unlabel_share_weight = normalize_weight(unlabel_share_weight)

        adv_loss = torch.zeros(1).to(self.device)
        adv_loss_separate = torch.zeros(1).to(self.device)

        tmp = self.l_weight[lb_idx] * nn.BCELoss(reduction="none")(l_domain_prob, torch.zeros_like(l_domain_prob))
        adv_loss += torch.mean(tmp, dim=0)
        tmp = self.u_weight[ulb_idx] * nn.BCELoss(reduction="none")(u_domain_prob, torch.ones_like(u_domain_prob))
        adv_loss += torch.mean(tmp, dim=0)

        self.l_weight[lb_idx] = label_share_weight
        self.u_weight[ulb_idx] = unlabel_share_weight
        # D'
        # tmp = cos_sim * nn.BCELoss(reduction="none")(domain_prob_separate_mix, torch.ones_like(domain_prob_separate_mix)*(1 - lam))
        tmp = cos_sim * (-1. * (1 - lam) * torch.log(domain_prob_separate_mix) - lam * torch.log(
            1 - domain_prob_separate_mix))
        adv_loss_separate += torch.mean(tmp, dim=0)
        adv_loss_separate += nn.BCELoss()(l_domain_prob_separate, torch.zeros_like(l_domain_prob_separate))
        adv_loss_separate += nn.BCELoss()(u_domain_prob_separate, torch.ones_like(u_domain_prob_separate))

        if self.it_total  > 100:
            w_ulb_logits = pseudo_label_calibration(w_ulb_logits, self.class_weight)

        # ramp up exp(-5(1 - t)^2)
        # coef = 1. * math.exp(-5 * (1 - min(self.it_total / (self.warmup*self.num_it_total), 1)) ** 2)
        # pseudo_label = torch.softmax(u_output.detach() / self.T, dim=-1)
        # max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        # mask = max_probs.ge(self.threshold).float()
        # ssl_loss = (Cross_Entropy(reduction='none')(s_u_output, targets_u) * mask).mean()* coef
        pseudo_label = torch.softmax(w_ulb_logits.detach() / self.T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()
        ssl_loss = (Cross_Entropy(reduction='none')(s_ulb_logits, targets_u) * mask).mean()*self.lambda_u
        # supervised loss
        cls_loss = Cross_Entropy(reduction='mean')(logits=lb_logits,targets=lb_y)

        adv_coef = 1. * math.exp(-5 * (1 - min(self.it_total / self.adv_warmup, 1)) ** 2)
        return cls_loss , ssl_loss , adv_coef , adv_loss , adv_loss_separate

    def init_optimizer(self):
        self._optimizer=copy.deepcopy(self.optimizer)
        self._discriminator_optimizer = copy.deepcopy(self.discriminator_optimizer)
        self._discriminator_optimizer_separate = copy.deepcopy(self.discriminator_optimizer_separate)
        if isinstance(self._optimizer,BaseOptimizer):
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in self._network.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self._network.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self._optimizer=self._optimizer.init_optimizer(params=grouped_parameters)

        if isinstance(self._discriminator_optimizer,BaseOptimizer):
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in self._discriminator.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self._discriminator.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self._discriminator_optimizer=self._discriminator_optimizer.init_optimizer(params=grouped_parameters)

        if isinstance(self._discriminator_optimizer_separate,BaseOptimizer):
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in self._discriminator_separate.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self._discriminator_separate.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self._discriminator_optimizer_separate=self._discriminator_optimizer_separate.init_optimizer(params=grouped_parameters)

    def init_scheduler(self):
        self._scheduler=copy.deepcopy(self.scheduler)
        if isinstance(self._scheduler,BaseScheduler):
            self._scheduler=self._scheduler.init_scheduler(optimizer=self._optimizer)

        self._discriminator_scheduler=copy.deepcopy(self.discriminator_scheduler)
        if isinstance(self._discriminator_scheduler,BaseScheduler):
            self._discriminator_scheduler=self._discriminator_scheduler.init_scheduler(optimizer=self._discriminator_optimizer)

        self._discriminator_scheduler_separate=copy.deepcopy(self.discriminator_scheduler_separate)
        if isinstance(self._discriminator_scheduler_separate,BaseScheduler):
            self._discriminator_scheduler_separate=self._discriminator_scheduler_separate.init_scheduler(optimizer=self._discriminator_optimizer_separate)



    def optimize(self,loss,*args,**kwargs):
        self._network.zero_grad()
        self._discriminator.zero_grad()
        self._discriminator_separate.zero_grad()
        loss.backward()
        self._optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()
        self._discriminator_optimizer.step()
        if self._discriminator_scheduler is not None:
            self._discriminator_scheduler.step()
        self._discriminator_optimizer_separate.step()
        if self._discriminator_scheduler_separate is not None:
            self._discriminator_scheduler_separate.step()
        if self.ema is not None:
            self.ema.update()


    def end_fit_epoch(self, *args, **kwargs):
        self.class_weight = compute_class_weight(self.l_weight, self.label_all, self.class_weight)

    @torch.no_grad()
    def estimate(self, X, idx=None, *args, **kwargs):
        _,outputs = self._network(X)
        return outputs


    def get_loss(self,train_result,*args,**kwargs):
        # lb_logits,lb_y,ulb_logits_1,ulb_logits_2=train_result
        # sup_loss = Cross_Entropy(reduction='mean')(lb_logits, lb_y)
        # _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        # unsup_loss = _warmup * Consistency(reduction='mean')(ulb_logits_1,ulb_logits_2.detach())
        # loss = Semi_Supervised_Loss(self.lambda_u)(sup_loss ,unsup_loss)
        cls_loss, ssl_loss, adv_coef, adv_loss, adv_loss_separate=train_result
        loss = cls_loss + ssl_loss + adv_coef * (adv_loss + adv_loss_separate)
        return loss

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)
