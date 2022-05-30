import copy
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from Semi_sklearn.Opitimizer.SemiOptimizer import SemiOptimizer
from Semi_sklearn.Scheduler.SemiScheduler import SemiScheduler
from sklearn.base import ClassifierMixin
from Semi_sklearn.utils import EMA
import torch
from Semi_sklearn.utils import cross_entropy
from Semi_sklearn.utils import class_status
from torch.nn import Softmax
from Semi_sklearn.utils import to_device
from Semi_sklearn.utils import Bn_Controller,one_hot
import math

# def fix_bn(m,train=False):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         if train:
#             m.train()
#         else:
#             m.eval()

class Fully(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
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
                 lambda_u=None,
                 mu=None,
                 ema_decay=None,
                 threshold=0.95,
                 num_classes=None,
                 tsa_schedule=None,
                 weight_decay=None,
                 T=0.4
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
        self.ema_decay=ema_decay
        self.lambda_u=lambda_u
        self.threshold=threshold
        # self.tsa_schedule=tsa_schedule
        self.num_classes=num_classes
        self.T=T
        self.weight_decay=weight_decay
        self.bn_controller = Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        # self._train_dataset.add_unlabeled_transform(copy.deepcopy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        # self._train_dataset.add_unlabeled_transform(self.strongly_augmentation,dim=1,x=1,y=0)

    def init_augmentation(self):
        # print(self._augmentation)
        if self._augmentation is not None:
            if isinstance(self._augmentation, dict):
                self.weakly_augmentation = self._augmentation['augmentation'] \
                    if 'augmentation' in self._augmentation.keys() \
                    else self._augmentation['weakly_augmentation']
                if 'strongly_augmentation' in self._augmentation.keys():
                    self.strongly_augmentation = self._augmentation['strongly_augmentation']
            elif isinstance(self._augmentation, (list, tuple)):
                self.weakly_augmentation = self._augmentation[0]
                if len(self._augmentation) > 1:
                    self.strongly_augmentation = self._augmentation[1]
            else:
                self.strongly_augmentation = copy.deepcopy(self._augmentation)
                self.weakly_augmentation=None

    def init_scheduler(self):
        if isinstance(self._scheduler,(list,tuple)):
            self._schedulerH=self._scheduler[0]
            self._schedulerF = self._scheduler[1]
        elif isinstance(self._scheduler,dict):
            self._schedulerH = self._scheduler['H'] if 'H' in self._scheduler.keys() \
                else self._scheduler['F']
            self._schedulerF = self._scheduler['F'] if 'F' in self._scheduler.keys() \
                else self._scheduler['H']
        else:
            self._schedulerH=self._scheduler
            self._schedulerF=copy.deepcopy(self._scheduler)

        if isinstance(self._schedulerH,SemiScheduler):
            self._schedulerH=self._schedulerH.init_scheduler(optimizer=self._optimizerH)
        if isinstance(self._schedulerF,SemiScheduler):
            self._schedulerF=self._schedulerF.init_scheduler(optimizer=self._optimizerF)

    def init_ema(self):
        if self.ema_decay is not None:
            if isinstance(self.ema_decay, (list, tuple)):
                self.ema_decayH = self.ema_decay[0]
                self.ema_decayF = self.ema_decay[1]
            elif isinstance(self.ema_decay, dict):
                self.ema_decayH = self.ema_decay['H'] if 'H' in self.ema_decay.keys() \
                    else self.ema_decay['F']
                self.ema_decayF = self.ema_decay['F'] if 'F' in self.ema_decay.keys() \
                    else self.ema_decay['H']
            else:
                self.ema_decayH = self.ema_decay
                self.ema_decayF = copy.deepcopy(self.ema_decay)
            self.emaH=EMA(model=self._networkH,decay=self.ema_decayH)
            self.emaH.register()
            self.emaF=EMA(model=self._networkF,decay=self.ema_decayF)
            self.emaF.register()
        else:
            self.emaH = None
            self.emaF = None

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_class
        self._networkH.zero_grad()
        self._networkH.train()
        self._networkF.zero_grad()
        self._networkF.train()

    def init_model(self):
        if isinstance(self._network,(list,tuple)):
            self._networkH=self._network[0]
            self._networkF = self._network[1]
        elif isinstance(self._network,dict):
            self._networkH = self._network['H'] if 'H' in self._network.keys() \
                else self._network['F']
            self._networkF = self._network['F'] if 'F' in self._network.keys() \
                else self._network['H']
        else:
            self._networkH=self._network
            self._networkF=copy.deepcopy(self._network)
        if self.device is not None and self.device is not 'cpu':
            torch.cuda.set_device(self.device)
        self._networkH=self._networkH.to(self.device)
        self._networkF = self._networkF.to(self.device)
        if self._parallel is not None:
            if isinstance(self._parallel, (list, tuple)):
                self._parallelH = self._parallel[0]
                self._parallelF = self._parallel[1]
            elif isinstance(self._parallel, dict):
                self._parallelH = self._parallel['H'] if 'H' in self._parallel.keys() \
                    else self._parallel['F']
                self._parallelF = self._parallel['F'] if 'F' in self._parallel.keys() \
                    else self._parallel['H']
            else:
                self._parallelH = self._parallel
                self._parallelF = copy.deepcopy(self._parallel)
            self._networkH=self._parallelH.init_parallel(self._networkH)
            self._networkF = self._parallelF.init_parallel(self._networkF)

    def init_optimizer(self):
        if isinstance(self._optimizer,(list,tuple)):
            self._optimizerH=self._optimizer[0]
            self._optimizerF = self._optimizer[1]
        elif isinstance(self._optimizer,dict):
            self._optimizerH = self._optimizer['H'] if 'H' in self._optimizer.keys() \
                else self._optimizer['F']
            self._optimizerF = self._optimizer['F'] if 'F' in self._optimizer.keys() \
                else self._optimizer['H']
        else:
            self._optimizerH=self._optimizer
            self._optimizerF=copy.deepcopy(self._optimizer)

        if isinstance(self._optimizerH,SemiOptimizer):
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in self._networkH.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self._networkH.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self._optimizerH=self._optimizerH.init_optimizer(params=grouped_parameters)

        if isinstance(self._optimizerF,SemiOptimizer):
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in self._networkF.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self._networkF.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self._optimizerF=self._optimizerF.init_optimizer(params=grouped_parameters)

    def train_batch_loop(self,valid_X=None,valid_y=None):
        for (lb_idx, lb_X, lb_y), (ulb_idx, ulb_X, _) in zip(self._labeled_dataloader, self._unlabeled_dataloader):
            if self.it_epoch >= self.num_it_epoch or self.it_total >= self.num_it_total:
                break

            self.start_batch_train()

            lb_idx = to_device(lb_idx,self.device)

            lb_X = to_device(lb_X,self.device)
            lb_y = to_device(lb_y,self.device)
            ulb_idx = to_device(ulb_idx,self.device)
            ulb_X  = to_device(ulb_X,self.device)

            lb_X = lb_X[0] if isinstance(lb_X, (list, tuple)) else lb_X
            lb_y = lb_y[0] if isinstance(lb_y, (list, tuple)) else lb_y
            ulb_X = ulb_X[0] if isinstance(ulb_X, (list, tuple)) else ulb_X

            num_unlabeled = ulb_X.shape[0]

            train_H_result = self.train_H(lb_X, lb_y)


            self.end_batch_train_H(train_H_result)

            self.bn_controller.freeze_bn(self._networkH)
            logits_ulb = self._networkH(ulb_X)
            self.bn_controller.unfreeze_bn(self._network)

            inputs=torch.cat([lb_X,ulb_X], dim=0)
            logits=torch.cat([one_hot(lb_y,self.num_classes,self.device),logits_ulb],dim=0)
            train_F_result = self.train_F(inputs,logits)

            self.end_batch_train_F(train_F_result)
            # self.end_batch_train(train_result)

            self.it_total += 1
            self.it_epoch += 1
            print(self.it_total,file=self.file)

            if valid_X is not None and self.eval_it is not None and self.it_total % self.eval_it == 0:
                self.evaluate(X=valid_X, y=valid_y,valid=True)

    def train_H(self,X,y):
        logits = self._networkH(X)
        return logits,y

    def train_F(self,X,y):
        logits = self._networkF(X)
        return logits,y


    def get_loss_H(self,train_result_H):
        logits,y=train_result_H
        loss = cross_entropy(logits, y, reduction='none').mean()
        return loss

    def get_loss_F(self,train_result_F):
        logits,y=train_result_F
        loss = cross_entropy(logits, y,use_hard_labels=False, reduction='none').mean()
        return loss

    def end_batch_train_H(self,train_H_result):
        loss = self.get_loss_H(train_H_result)
        self.optimize_H(loss)

    def end_batch_train_F(self,train_F_result):
        loss = self.get_loss_F(train_F_result)
        self.optimize_F(loss)

    def optimize_H(self,loss):
        self._optimizerH.zero_grad()
        loss.backward()
        self._optimizerH.step()
        if self._schedulerH is not None:
            self._schedulerH.step()
        if self.emaH is not None:
            self.emaH.update()

    def optimize_F(self,loss):
        self._optimizerF.zero_grad()
        loss.backward()
        self._optimizerF.step()
        if self._schedulerF is not None:
            self._schedulerF.step()
        if self.emaF is not None:
            self.emaF.update()


    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)


