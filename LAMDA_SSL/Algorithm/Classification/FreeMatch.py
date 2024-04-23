import copy
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
import LAMDA_SSL.Config.FixMatch as config
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
from LAMDA_SSL.Loss.Semi_Supervised_Loss import Semi_Supervised_Loss
from LAMDA_SSL.utils import class_status

import torch

class FreeMatch(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 ema_p=0.999,
                 use_DA=False,
                 num_classes=None,
                 threshold=config.threshold,
                 lambda_u=config.lambda_u,
                 T=config.T,
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
        self.num_classes=num_classes
        self.ema_p=ema_p
        self.use_DA=use_DA
        self.lambda_u=lambda_u
        self.threshold=threshold
        self.T=T
        self.weight_decay=weight_decay
        self._estimator_type=ClassifierMixin._estimator_type
    
    @torch.no_grad()
    def update_prob_t(self, lb_probs, ulb_probs):
        ulb_prob_t = ulb_probs.mean(0)
        self.ulb_prob_t = self.ema_p * self.ulb_prob_t + (1 - self.ema_p) * ulb_prob_t

        lb_prob_t = lb_probs.mean(0)
        self.lb_prob_t = self.ema_p * self.lb_prob_t + (1 - self.ema_p) * lb_prob_t

        max_probs, max_idx = ulb_probs.max(dim=-1)
        prob_max_mu_t = torch.mean(max_probs)
        prob_max_var_t = torch.var(max_probs, unbiased=True)
        self.prob_max_mu_t = self.ema_p * self.prob_max_mu_t + (1 - self.ema_p) * prob_max_mu_t.item()
        self.prob_max_var_t = self.ema_p * self.prob_max_var_t + (1 - self.ema_p) * prob_max_var_t.item()
    
    @torch.no_grad()
    def calculate_mask(self, probs):
        max_probs, max_idx = probs.max(dim=-1)
        mu = self.prob_max_mu_t
        var = self.prob_max_var_t
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / 4)))
        return mask.detach(), max_idx.detach()
    
    def init_transform(self):
        self._train_dataset.add_unlabeled_transform(copy.copy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.strong_augmentation,dim=1,x=1,y=0)

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_classes
        self.p_model = (torch.ones(self.num_classes) / self.num_classes).to(self.device)
        self.label_hist = (torch.ones(self.num_classes) / self.num_classes).to(self.device) 
        self.time_p = self.p_model.mean()
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y
        w_ulb_X,s_ulb_X=ulb_X[0],ulb_X[1]
        batch_size = lb_X.shape[0]
        inputs=torch.cat((lb_X, w_ulb_X, s_ulb_X))
        logits = self._network(inputs)
        lb_logits = logits[:batch_size]
        w_ulb_logits, s_ulb_logits = logits[batch_size:].chunk(2)
        train_result=(lb_logits,lb_y,w_ulb_logits, s_ulb_logits)
        return train_result

    @torch.no_grad()
    def distribution_alignment(self, probs):
        # da
        probs = probs * self.lb_prob_t / self.ulb_prob_t
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs.detach()

    def cal_time_p_and_p_model(self,logits_x_ulb_w, time_p, p_model, label_hist):
        prob_w = torch.softmax(logits_x_ulb_w, dim=1) 
        max_probs, max_idx = torch.max(prob_w, dim=-1)
        if time_p is None:
            time_p = max_probs.mean()
        else:
            time_p = time_p * 0.999 +  max_probs.mean() * 0.001
        if p_model is None:
            p_model = torch.mean(prob_w, dim=0)
        else:
            p_model = p_model * 0.999 + torch.mean(prob_w, dim=0) * 0.001
        if label_hist is None:
            label_hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype) 
            label_hist = label_hist / label_hist.sum()
        else:
            hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype) 
            label_hist = label_hist * 0.999 + (hist / hist.sum()) * 0.001
        return time_p,p_model,label_hist

    def get_loss(self,train_result,*args,**kwargs):
        lb_logits, lb_y, w_ulb_logits, s_ulb_logits = train_result
        probs_x_lb = torch.softmax(lb_logits.detach(), dim=-1)
        sup_loss=Cross_Entropy(reduction='mean')(logits=lb_logits,targets=lb_y)
        self.time_p, self.p_model, self.label_hist = self.cal_time_p_and_p_model(w_ulb_logits, self.time_p, self.p_model, self.label_hist)
        pseudo_label = torch.softmax(w_ulb_logits.detach(), dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        p_cutoff = self.time_p
        p_model_cutoff = self.p_model / torch.max(self.p_model,dim=-1)[0]
        threshold = p_cutoff * p_model_cutoff[max_idx]
        mask = max_probs.ge(threshold)

        # mask, targets_u= self.calculate_mask(probs_x_ulb_w)
        
        # pseudo_label = torch.softmax(w_ulb_logits.detach() / self.T, dim=-1)
        # max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        # mask = max_probs.ge(self.threshold).float()
        unsup_loss = (Cross_Entropy(reduction='none')(s_ulb_logits, max_idx) * mask).mean()
        loss=Semi_Supervised_Loss(lambda_u =self.lambda_u)(sup_loss,unsup_loss)
        return loss

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)
