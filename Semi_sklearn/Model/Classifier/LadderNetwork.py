import copy

from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from Semi_sklearn.Network.Ladder import Ladder
from torch.autograd import Variable
from sklearn.base import ClassifierMixin
import torch
import torch.nn as nn
from Semi_sklearn.Opitimizer.SemiOptimizer import SemiOptimizer
class Ladder_Network(InductiveEstimator,SemiDeepModelMixin,ClassifierMixin):
    def __init__(self,
                 dim_in,
                 num_class,
                 noise_std=0.2,
                 lambda_u=[0.1, 0.1, 0.1, 0.1, 0.1, 10., 1000.],
                 encoder_sizes=[1000, 500, 250, 250, 250],
                 encoder_activations=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
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
        network=Ladder(encoder_sizes=encoder_sizes, encoder_activations=encoder_activations,
                  noise_std=noise_std,dim_in=dim_in,n_class=num_class,device=device) if network is None else network
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
        self.dim_in=dim_in
        self.num_class=num_class
        self.noise_std = noise_std
        self.lambda_u = lambda_u
        self.encoder_sizes = encoder_sizes
        self.encoder_activations = encoder_activations
        self._estimator_type = ClassifierMixin._estimator_type

    def init_optimizer(self):
        if isinstance(self._optimizer,SemiOptimizer):
            self._optimizer=self._optimizer.init_optimizer(params=self._network.parameters())

    def train(self,lb_X=None,lb_y=None,ulb_X=None,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X = lb_X[0] if isinstance(lb_X,(list,tuple)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(list,tuple)) else lb_y
        ulb_X=ulb_X[0]if isinstance(ulb_X,(list,tuple)) else ulb_X
        lb_X=lb_X*1/255.
        ulb_X = ulb_X * 1 / 255.
        lb_X=lb_X.view(lb_X.shape[0],-1)
        ulb_X = ulb_X.view(ulb_X.shape[0], -1)
        lb_X= Variable(lb_X, requires_grad=False)
        lb_y = Variable(lb_y, requires_grad=False)
        ulb_X = Variable(ulb_X)

        # do a noisy pass for labelled data
        output_noise_labeled = nn.Softmax(dim=-1)(self._network.forward_encoders_noise(lb_X))

        # do a noisy pass for unlabelled_data
        output_noise_unlabeled = nn.Softmax(dim=-1)(self._network.forward_encoders_noise(ulb_X))
        tilde_z_layers_unlabeled = self._network.get_encoders_tilde_z(reverse=True)

        # do a clean pass for unlabelled data
        output_clean_unlabeled = nn.Softmax(dim=-1)(self._network.forward_encoders_clean(ulb_X))
        z_pre_layers_unlabeled = self._network.get_encoders_z_pre(reverse=True)
        z_layers_unlabeled = self._network.get_encoders_z(reverse=True)

        tilde_z_bottom_unlabeled = self._network.get_encoder_tilde_z_bottom()

        # pass through decoders
        hat_z_layers_unlabeled = self._network.forward_decoders(tilde_z_layers_unlabeled,
                                                          output_noise_unlabeled,
                                                          tilde_z_bottom_unlabeled)

        z_pre_layers_unlabeled.append(ulb_X)
        z_layers_unlabeled.append(ulb_X)

        # batch normalize using mean, var of z_pre
        bn_hat_z_layers_unlabeled = self._network.decoder_bn_hat_z_layers(hat_z_layers_unlabeled, z_pre_layers_unlabeled)
        return output_noise_labeled, lb_y, z_layers_unlabeled, bn_hat_z_layers_unlabeled

    def get_loss(self,train_result,*args,**kwargs):
        output_noise_labeled, lb_y, z_layers_unlabeled, bn_hat_z_layers_unlabeled=train_result
        loss_supervised = torch.nn.CrossEntropyLoss()
        loss_unsupervised = torch.nn.MSELoss()
        cost_supervised = loss_supervised(output_noise_labeled, lb_y)
        cost_unsupervised = 0.
        for cost_lambda, z, bn_hat_z in zip(self.lambda_u, z_layers_unlabeled, bn_hat_z_layers_unlabeled):
            c = cost_lambda * loss_unsupervised.forward(bn_hat_z, z)
            cost_unsupervised += c
        result = cost_supervised + cost_unsupervised
        # result=cost_unsupervised
        return result

    def optimize(self,loss,*args,**kwargs):
        self._network.zero_grad()
        loss.backward()
        self._optimizer.step()



    def end_epoch(self):
        self._scheduler.step()

    @torch.no_grad()
    def estimate(self, X, idx=None, *args, **kwargs):
        X=X*1/255.
        _X=X.view(X.shape[0],-1)
        outputs = self._network(_X)
        return outputs

    def predict(self,X=None,valid=None):
        return SemiDeepModelMixin.predict(self,X=X,valid=valid)