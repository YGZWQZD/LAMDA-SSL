import copy
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
import LAMDA_SSL.Config.FixMatch as config
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
from LAMDA_SSL.Loss.Semi_Supervised_Loss import Semi_Supervised_Loss
from LAMDA_SSL.utils import class_status
import torch
import torch.nn.functional as F
import numpy as np

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

class Fix_A_Step(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
    			 num_classes=None,
    			 warmup=0,
    			 alpha=0.75,
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
        self.lambda_u=lambda_u
        self.alpha=alpha
        self.warmup=warmup
        self.threshold=threshold
        self.num_classes=num_classes
        self.T=T
        self.weight_decay=weight_decay
        self._estimator_type=ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_unlabeled_transform(copy.copy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_unlabeled_transform(copy.copy(self.train_dataset.unlabeled_transform),dim=0,x=2)
        self._train_dataset.add_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=1,y=0)
        self._train_dataset.add_unlabeled_transform(self.strong_augmentation,dim=1,x=2,y=0)

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_classes
        self._network.zero_grad()
        self._network.train()
        
    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y
        w_ulb_X_1, w_ulb_X_2, s_ulb_X=ulb_X[0],ulb_X[1],ulb_X[2]
        batch_size = lb_X.shape[0]
        with torch.no_grad():
            u_output1 = self._network(w_ulb_X_1)
            u_output2 = self._network(w_ulb_X_2)
            p = (torch.softmax(u_output1, dim=1) + torch.softmax(u_output2, dim=1)) / 2
            pt = p**(1/self.T)
            u_targets = pt/pt.sum(dim=1, keepdim=True)
            u_targets = u_targets.detach()
        lb_y = torch.zeros(batch_size, self.num_classes).to(self.device).scatter_(1, lb_y.view(-1,1).long(), 1)
        Augment_combined_inputs = torch.cat([lb_X, w_ulb_X_1, w_ulb_X_2], dim=0)
        Augment_combined_labels = torch.cat([lb_y, u_targets, u_targets], dim=0)
        
        l = np.random.beta(self.alpha, self.alpha)
        l = max(l, 1-l)
        
        idx = torch.randperm(Augment_combined_inputs.size(0))

        input_a, input_b = Augment_combined_inputs, Augment_combined_inputs[idx]
        target_a, target_b = Augment_combined_labels, Augment_combined_labels[idx]
        
        mixed_input = l * input_a + (1-l) * input_b        
        mixed_target = l * target_a + (1-l) * target_b
        
        mixed_labeled_input = torch.split(mixed_input, batch_size)[0]
        mixed_labeled_target = torch.split(mixed_target, batch_size)[0]        
        
        inputs = interleave(torch.cat((mixed_labeled_input, w_ulb_X_1, s_ulb_X)), 2*self.mu+1).to(self.device)
        
        logits = self._network(inputs)
        logits = de_interleave(logits, 2*self.mu+1)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        
        del logits
        
        labeledtrain_loss = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_labeled_target, dim=1))

        labeledtrain_loss.backward(retain_graph=True)

        labeled_grads = []
        for name, param in self._network.named_parameters():
            try:
                labeled_grads.append(param.grad.view(-1))
            except:
                continue
            
        labeled_grads = torch.cat(labeled_grads)
# 
        self._network.zero_grad()
        
        pseudo_label = torch.softmax(logits_u_w.detach()/self.T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()
        
        unlabeledtrain_loss = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
        unlabeledtrain_loss.backward(retain_graph=True)

        unlabeled_grads = []
        for name, param in self._network.named_parameters():
            try:
                unlabeled_grads.append(param.grad.view(-1))
            except:
                continue
                
            
        unlabeled_grads = torch.cat(unlabeled_grads)

        self._network.zero_grad()
        
        gradient_dot = torch.dot(labeled_grads, unlabeled_grads)


        current_lambda_u = self.lambda_u
        

        if self.it_total>= float(self.warmup*self.num_it_total):
            if gradient_dot<0:
                # gradient_dot_sign_prob.update(-1)
                # gradient_dot_sign_this_epoch.append(-1)
                loss = labeledtrain_loss
            else:
                # gradient_dot_sign_prob.update(1)
                # gradient_dot_sign_this_epoch.append(1)
                loss = labeledtrain_loss + current_lambda_u * unlabeledtrain_loss
        else:
            loss = labeledtrain_loss + current_lambda_u * unlabeledtrain_loss
        return loss

    def get_loss(self,train_result,*args,**kwargs):
        loss = train_result
        return loss

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)