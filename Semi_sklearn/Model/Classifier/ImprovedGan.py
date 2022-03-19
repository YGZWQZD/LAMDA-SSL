import copy

import numpy as np
import torch.nn.functional as F

from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from Semi_sklearn.Opitimizer.SemiOptimizer import SemiOptimizer
import Semi_sklearn.Network.ImprovedGan as ImGan
import random
from sklearn.base import ClassifierMixin
import torch
from Semi_sklearn.utils import one_hot
from torch.autograd import Variable
from Semi_sklearn.utils import to_device
from Semi_sklearn.Base.GeneratorMixin import GeneratorMixin

class ImprovedGan(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin,GeneratorMixin):
    def __init__(self,
                 dim_in,
                 num_class,
                 dim_z,
                 dim_hidden,
                 num_labeled=None,
                 train_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 train_dataloader=None,
                 labeled_dataloader=None,
                 unlabeled_dataloader=None,
                 valid_dataloader=None,
                 test_dataloader=None,
                 augmentation=None,
                 epoch=1,
                 network=None,
                 num_it_epoch=None,
                 num_it_total=None,
                 eval_epoch=None,
                 eval_it=None,
                 mu=None,
                 optimizer=None,
                 weight_decay=5e-4,
                 lambda_u=1.0,
                 ema_decay=None,
                 scheduler=None,
                 device=None,
                 evaluation=None,
                 train_sampler=None,
                 labeled_sampler=None,
                 unlabeled_sampler=None,
                 train_batch_sampler=None,
                 labeled_batch_sampler=None,
                 unlabeled_batch_sampler=None,
                 valid_sampler=None,
                 valid_batch_sampler=None,
                 test_sampler=None,
                 test_batch_sampler=None,
                 parallel=None):
        network=ImGan.ImprovedGAN( dim_in = dim_in, output_dim = num_class,z_dim=dim_z) if network is None else network
        SemiDeepModelMixin.__init__(self, train_dataset=train_dataset,
                                    valid_dataset=valid_dataset,
                                    test_dataset=test_dataset,
                                    train_dataloader=train_dataloader,
                                    labeled_dataloader=labeled_dataloader,
                                    unlabeled_dataloader=unlabeled_dataloader,
                                    valid_dataloader=valid_dataloader,
                                    test_dataloader=test_dataloader,
                                    augmentation=augmentation,
                                    network=network,
                                    train_sampler=train_sampler,
                                    labeled_sampler=labeled_sampler,
                                    unlabeled_sampler=unlabeled_sampler,
                                    train_batch_sampler=train_batch_sampler,
                                    labeled_batch_sampler=labeled_batch_sampler,
                                    unlabeled_batch_sampler=unlabeled_batch_sampler,
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
                                    evaluation=evaluation,
                                    parallel=parallel
                                    )
        self.dim_hidden=dim_hidden
        self.dim_z=dim_z
        self.dim_in=dim_in
        self.num_class=num_class
        self.lambda_u=lambda_u
        self.num_labeled=num_labeled
        self._estimator_type = [GeneratorMixin._estimator_type,ClassifierMixin._estimator_type]

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
            # print(lb_X.max())
            num_unlabeled = ulb_X.shape[0]
            lb_X=lb_X*1/255.
            ulb_X = ulb_X * 1 / 255.
            ulb_X_1, ulb_X_2 = ulb_X[:num_unlabeled // 2], ulb_X[num_unlabeled // 2:]

            train_D_result = self.train_D(lb_X, lb_y, ulb_X_1)

            self.end_batch_train_D(train_D_result)

            train_G_result = self.train_G(ulb_X_2)

            self.end_batch_train_G(train_G_result)
            # self.end_batch_train(train_result)

            self.it_total += 1
            self.it_epoch += 1
            print(self.it_total)

            if valid_X is not None and self.eval_it is not None and self.it_total % self.eval_it == 0:
                self.evaluate(X=valid_X, y=valid_y,valid=True)

    def init_optimizer(self):
        if isinstance(self._optimizer,(list,tuple)):
            self._optimizerG=self._optimizer[0]
            self._optimizerD = self._optimizer[1]
        elif isinstance(self._optimizer,dict):
            self._optimizerG = self._optimizer['Generator'] if 'Generator' in self._optimizer.keys() \
                else self._optimizer['Generation']
            self._optimizerD = self._optimizer['Discriminator'] if 'Discriminator' in self._optimizer.keys() \
                else self._optimizer['Discrimination']
        else:
            self._optimizerG=self._optimizer
            self._optimizerD=copy.deepcopy(self._optimizer)

        if isinstance(self._optimizerG,SemiOptimizer):
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in self._network.G.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self._network.G.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self._optimizerG=self._optimizerG.init_optimizer(params=grouped_parameters)

        if isinstance(self._optimizerD,SemiOptimizer):
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in self._network.D.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self._network.D.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self._optimizerD=self._optimizerD.init_optimizer(params=grouped_parameters)

    def init_augmentation(self):
        if self._augmentation is not None:
            if isinstance(self._augmentation, dict):
                self.to_image = self._augmentation['augmentation'] \
                    if 'augmentation' in self._augmentation.keys() \
                    else self._augmentation['To_image']
            elif isinstance(self._augmentation, (list, tuple)):
                self.to_image = self._augmentation[0]
            else:
                self.to_image = copy.deepcopy(self._augmentation)

    def init_transform(self):
        self._train_dataset.add_transform(self.to_image, dim=1, x=0, y=0)
        self._train_dataset.add_unlabeled_transform(self.to_image, dim=1, x=0, y=0)
        self._test_dataset.add_transform(self.to_image, dim=1, x=0, y=0)
        self._valid_dataset.add_transform(self.to_image, dim=1, x=0, y=0)

    def get_loss_D(self,train_result_D):
        output_label, lb_y,output_unlabel, output_fake=train_result_D
        logz_label, logz_unlabel, logz_fake = self.log_sum_exp(output_label), \
                                              self.log_sum_exp(output_unlabel), \
                                              self.log_sum_exp(output_fake) # log ∑e^x_i
        prob_label = torch.gather(output_label, 1, lb_y.unsqueeze(1)) # log e^x_label = x_label
        loss_supervised = -torch.mean(prob_label) + torch.mean(logz_label)
        loss_unsupervised = 0.5 * (-torch.mean(logz_unlabel) + torch.mean(F.softplus(logz_unlabel))  + # real_data: log Z/(1+Z)
                            torch.mean(F.softplus(logz_fake)))
        loss=loss_supervised+self.lambda_u*loss_unsupervised
        return loss

    def get_loss_G(self,train_result_G):
        mom_fake, mom_unlabel=train_result_G
        loss_fm = torch.mean((mom_fake - mom_unlabel) ** 2)
        loss =loss_fm
        return loss

    def end_batch_train_D(self,train_result_D):
        loss = self.get_loss_D(train_result_D)
        self.optimize_D(loss)

    def end_batch_train_G(self,train_result_G):
        loss = self.get_loss_G(train_result_G)
        self.optimize_G(loss)

    def optimize_D(self,loss):
        self._optimizerD.zero_grad()
        loss.backward()
        self._optimizerD.step()
        if self._scheduler is not None:
            self._scheduler.step()
        if self.ema is not None:
            self.ema.update()


    def optimize_G(self,loss):
        self._optimizerG.zero_grad()
        self._optimizerD.zero_grad()
        loss.backward()
        self._optimizerG.step()
        if self._scheduler is not None:
            self._scheduler.step()
        if self.ema is not None:
            self.ema.update()

    def log_sum_exp(self,x, axis=1):
        m = torch.max(x, dim=1)[0]
        return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))

    def train_D(self,X,y,unlabeled_X):
        X=X.view(X.shape[0],-1)
        unlabeled_X=unlabeled_X.view(unlabeled_X.shape[0],-1)
        output_label=self._network.D(X)
        output_unlabel=self._network.D(unlabeled_X)
        fake_X=self._network.G(unlabeled_X.size()[0]).view(unlabeled_X.size()).detach()
        # print(fake_X.shape)
        # print(unlabeled_X.shape)
        output_fake =self._network.D(fake_X)
        return output_label, y, output_unlabel, output_fake

    def train_G(self, unlabeled_X):
        unlabeled_X = unlabeled_X.view(unlabeled_X.shape[0], -1)
        fake = self._network.G(unlabeled_X.size()[0]).view(unlabeled_X.size())
        output_fake = self._network.D(fake)
        mom_fake=self._network.D.feature

        output_unlabeled = self._network.D(Variable(unlabeled_X))
        mom_unlabeled=self._network.D.feature

        mom_fake = torch.mean(mom_fake, dim = 0)
        mom_unlabel = torch.mean(mom_unlabeled, dim = 0)

        # self.Goptim.zero_grad()
        # self.Doptim.zero_grad()
        # loss.backward()
        # self.Goptim.step()
        return mom_fake,mom_unlabel









    def optimize(self,*args,**kwargs):
        self._optimizer.step()
        self._network.zero_grad()

    @torch.no_grad()
    def estimate(self, X, idx=None, *args, **kwargs):
        X=X.view(X.shape[0],-1)
        X=X*1/255.
        outputs = self._network(X)
        return outputs

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)

    def generate(self,num,z=None):

        z = Variable(torch.randn(num, self.dim_z).to(self.device)) if z is None else z

        return self._network.G(num,z)