import copy
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from LAMDA_SSL.Network.Ladder import Ladder
from torch.autograd import Variable
from sklearn.base import ClassifierMixin
import torch
import torch.nn as nn
from LAMDA_SSL.Opitimizer.BaseOptimizer import BaseOptimizer
import LAMDA_SSL.Config.LadderNetwork as config
from LAMDA_SSL.utils import class_status
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
from LAMDA_SSL.Loss.MSE import MSE

class Ladder_Network(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 dim_in=config.dim_in,
                 num_classes=config.num_classes,
                 noise_std=config.noise_std,
                 lambda_u=config.lambda_u,
                 dim_encoder=config.dim_encoder,
                 encoder_activations=config.encoder_activations,
                 epoch=config.epoch,
                 num_it_epoch=config.num_it_epoch,
                 num_it_total=config.num_it_total,
                 eval_epoch=config.eval_epoch,
                 eval_it=config.eval_it,
                 mu=config.mu,
                 weight_decay=config.weight_decay,
                 ema_decay=config.ema_decay,
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
                 network=config.network,
                 optimizer=config.optimizer,
                 scheduler=config.scheduler,
                 device=config.device,
                 evaluation=config.evaluation,
                 train_sampler=config.train_sampler,
                 labeled_sampler=config.labeled_sampler,
                 unlabeled_sampler=config.unlabeled_sampler,
                 train_batch_sampler=config.train_batch_sampler,
                 labeled_batch_sampler=config.train_batch_sampler,
                 unlabeled_batch_sampler=config.unlabeled_batch_sampler,
                 valid_sampler=config.valid_sampler,
                 valid_batch_sampler=config.valid_batch_sampler,
                 test_sampler=config.test_sampler,
                 test_batch_sampler=config.test_batch_sampler,
                 parallel=config.parallel,
                 file=config.file,
                 verbose=config.verbose,
                 ):
        # >> Parameter:
        # >> - dim_in: The dimension of a single instance.
        # >> - num_classes: The number of classes.
        # >> - noise_std: The noise level of each layer of the discriminator.
        # >> - lambda_u: The proportion of consistency loss of each layer in LadderNetwork.
        # >> - encoder_sizes: The dimension of each layer of the encoder.
        # >> - encoder_activations: The activation function of each layer of the encoder.
        DeepModelMixin.__init__(self, train_dataset=train_dataset,
                                labeled_dataset=labeled_dataset,
                                unlabeled_dataset=unlabeled_dataset,
                                valid_dataset=valid_dataset,
                                test_dataset=test_dataset,
                                train_dataloader=train_dataloader,
                                labeled_dataloader=labeled_dataloader,
                                unlabeled_dataloader=unlabeled_dataloader,
                                valid_dataloader=valid_dataloader,
                                test_dataloader=test_dataloader,
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
                                parallel=parallel,
                                file=file,
                                verbose=verbose)
        self.dim_in=dim_in
        self.num_classes=num_classes
        self.noise_std = noise_std
        self.lambda_u = lambda_u
        self.dim_encoder = dim_encoder
        self.encoder_activations = encoder_activations
        self._estimator_type = ClassifierMixin._estimator_type

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_classes
        self.dim_in=self.dim_in if self.dim_in is not None else \
            self._train_dataset.labeled_dataset.X.shape[1:]
        if self.network is None:
            self.network=Ladder(dim_encoder=self.dim_encoder, encoder_activations=self.encoder_activations,
                      noise_std=self.noise_std,dim_in=self.dim_in,num_classes=self.num_classes,device=self.device)
            self._network=copy.deepcopy(self.network)
            self.init_model()
            self.init_ema()
            self.init_optimizer()
            self.init_scheduler()
        self._network.zero_grad()
        self._network.train()

    def init_optimizer(self):
        if isinstance(self._optimizer,BaseOptimizer):
            self._optimizer=self._optimizer.init_optimizer(params=self._network.parameters())

    def train(self,lb_X=None,lb_y=None,ulb_X=None,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X = lb_X[0] if isinstance(lb_X,(list,tuple)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(list,tuple)) else lb_y
        ulb_X=ulb_X[0]if isinstance(ulb_X,(list,tuple)) else ulb_X
        # lb_X=lb_X*1/255.
        # ulb_X = ulb_X * 1 / 255.
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
        cost_supervised = Cross_Entropy(reduction='mean')(output_noise_labeled, lb_y)
        cost_unsupervised = 0.
        for cost_lambda, z, bn_hat_z in zip(self.lambda_u, z_layers_unlabeled, bn_hat_z_layers_unlabeled):
            c = cost_lambda * MSE(reduction='mean')(bn_hat_z, z)
            cost_unsupervised += c
        result = cost_supervised + cost_unsupervised
        return result

    def optimize(self,loss,*args,**kwargs):
        self._network.zero_grad()
        loss.backward()
        self._optimizer.step()



    def end_fit_epoch(self):
        self._scheduler.step()

    @torch.no_grad()
    def estimate(self, X, idx=None, *args, **kwargs):
        _X=X.view(X.shape[0],-1)
        outputs = self._network(_X)
        return outputs

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)