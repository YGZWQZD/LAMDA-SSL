import copy

import numpy as np
import torch.nn.functional as F
from Semi_sklearn.Base.GeneratorMixin import GeneratorMixin
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
import Semi_sklearn.Network.SSVAE as VAE
import random
from sklearn.base import ClassifierMixin
import torch
from Semi_sklearn.utils import one_hot
from torch.autograd import Variable

class SSVAE(InductiveEstimator,SemiDeepModelMixin,GeneratorMixin,ClassifierMixin):
    def __init__(self,
                 dim_in,
                 num_class,
                 dim_z,
                 dim_hidden,
                 alpha,
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
        network=VAE.SSVAE( dim_in=dim_in,num_class=num_class,dim_z=dim_z,dim_hidden=dim_hidden,device=device) if network is None else network
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
        self.alpha=alpha
        self.num_labeled=num_labeled
        self._estimator_type = [GeneratorMixin._estimator_type,ClassifierMixin._estimator_type]

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

    def loss_components_fn(self,x, y, z, p_y, p_z, p_x_yz, q_z_xy):
        # SSL paper eq 6 for an given y (observed or enumerated from q_y)

        return - p_x_yz.log_prob(x).sum(1) \
               - p_y.log_prob(y) \
               - p_z.log_prob(z).sum(1) \
               + q_z_xy.log_prob(z).sum(1)

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        # print(lb_X)
        lb_X = lb_X[0] if isinstance(lb_X,(list,tuple)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(list,tuple)) else lb_y
        ulb_X=ulb_X[0]if isinstance(ulb_X,(list,tuple)) else ulb_X

        lb_X=lb_X.view(lb_X.shape[0],-1)
        ulb_X = ulb_X.view(ulb_X.shape[0], -1)

        lb_q_y = self._network.encode_y(lb_X)
        ulb_q_y = self._network.encode_y(ulb_X)

        lb_y = one_hot(lb_y, self.num_class,self.device).to(self.device)

        lb_q_z_xy = self._network.encode_z(lb_X, lb_y)

        lb_z = lb_q_z_xy.rsample()
        lb_p_x_yz = self._network.decode(lb_y, lb_z)
        # print(lb_p_x_yz)

        lb_p_y=self._network.p_y
        lb_p_z = self._network.p_z




        ulb_q_z_xy_list = []
        ulb_p_x_yz_list = []
        ulb_z_list=[]
        ulb_p_y=self._network.p_y
        ulb_p_z=self._network.p_z

        for ulb_y in ulb_q_y.enumerate_support():
            ulb_q_z_xy = self._network.encode_z(ulb_X, ulb_y)
            ulb_z = ulb_q_z_xy.rsample()
            ulb_p_x_yz = self._network.decode(ulb_y, ulb_z)
            ulb_z_list.append(ulb_z)
            ulb_q_z_xy_list.append(ulb_q_z_xy)
            ulb_p_x_yz_list.append(ulb_p_x_yz)




        return lb_X, lb_y, lb_z, lb_p_y, lb_p_z, lb_q_y, lb_p_x_yz, lb_q_z_xy, ulb_X,ulb_z_list,ulb_p_y,ulb_p_z, ulb_q_y,ulb_p_x_yz_list,ulb_q_z_xy_list


    def get_loss(self,train_result,*args,**kwargs):
        lb_X, lb_y, lb_z, lb_p_y, lb_p_z, lb_q_y, lb_p_x_yz, lb_q_z_xy, ulb_X,ulb_z_list,ulb_p_y,ulb_p_z, ulb_q_y,ulb_p_x_yz_list,ulb_q_z_xy_list=train_result
        sup_loss = self.loss_components_fn(lb_X, lb_y, lb_z, lb_p_y, lb_p_z, lb_p_x_yz, lb_q_z_xy)
        # print(sup_loss)
        # print(lb_q_y.log_prob(lb_y))
        # num_labeled=self.num_labeled if self.num_labeled is None else self._train_dataset.labeled_dataset.__len__()
        sup_loss=sup_loss.mean(0)
        probs=F.softmax(lb_q_y.probs,dim=-1)
        cls_loss=- self.alpha  * lb_q_y.log_prob(lb_y)
        cls_loss=cls_loss.mean(0)


        unsup_loss = - ulb_q_y.entropy()
        idx=0
        for ulb_y in ulb_q_y.enumerate_support():
            # print(ulb_z_list)
            L_xy = self.loss_components_fn(ulb_X, ulb_y, ulb_z_list[idx], ulb_p_y, ulb_p_z, ulb_p_x_yz_list[idx], ulb_q_z_xy_list[idx])
            unsup_loss += ulb_q_y.log_prob(ulb_y).exp() * L_xy
            idx+=1
        unsup_loss=unsup_loss.mean(0)
        # print(unsup_loss)
        result=sup_loss+unsup_loss+cls_loss
        return result



    def optimize(self,loss,*args,**kwargs):
        self._network.zero_grad()
        loss.backward()
        self._optimizer.step()

    @torch.no_grad()
    def estimate(self, X, idx=None, *args, **kwargs):
        X=X.view(X.shape[0],-1)
        outputs = self._network(X)
        return outputs

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)

    def generate(self,num,z=None,x=None,y=None):
        if y is not None:
            y = one_hot(y,self.num_class,self.device).to(self.device)
        if x is not None and y is None:
            y=self._network.encode_y(x)
        if x is not None and y is not None and z is None:
            z = self._network.encode_z(x,y)
        z = Variable(torch.randn(num, self.dim_z).to(self.device)) if z is None else z
        y = one_hot(Variable(torch.LongTensor([random.randrange(self.num_class) for _ in range(num)]).to(self.device)),nClass=self.num_class,device=self.device)

        return self._network.decode(z,y)
