import copy
import torch.nn.functional as F
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
import LAMDA_SSL.Network.SSVAE as VAE
import random
from sklearn.base import ClassifierMixin
import torch
from LAMDA_SSL.utils import one_hot
from torch.autograd import Variable
import LAMDA_SSL.Config.SSVAE as config
from LAMDA_SSL.utils import class_status

class SSVAE(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 alpha=config.alpha,
                 num_labeled=config.num_labeled,
                 dim_in=config.dim_in,
                 num_classes=config.num_classes,
                 dim_z=config.dim_z,
                 dim_hidden_de=config.dim_hidden_de,
                 dim_hidden_en_y=config.dim_hidden_en_y,
                 dim_hidden_en_z=config.dim_hidden_en_z,
                 activations_de=config.activations_de,
                 activations_en_y=config.activations_en_y,
                 activations_en_z=config.activations_en_z,
                 mu=config.mu,
                 weight_decay=config.weight_decay,
                 ema_decay=config.ema_decay,
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
                 labeled_sampler=config.labeled_sampler,
                 unlabeled_sampler=config.unlabeled_sampler,
                 train_batch_sampler=config.train_batch_sampler,
                 labeled_batch_sampler=config.labeled_batch_sampler,
                 unlabeled_batch_sampler=config.unlabeled_batch_sampler,
                 valid_sampler=config.valid_sampler,
                 valid_batch_sampler=config.valid_batch_sampler,
                 test_sampler=config.test_sampler,
                 test_batch_sampler=config.test_batch_sampler,
                 network=config.network,
                 optimizer=config.optimizer,
                 scheduler=config.scheduler,
                 evaluation=config.evaluation,
                 parallel=config.parallel,
                 file=config.file,
                 verbose=config.verbose):
        # >> Parameter
        # >> - alpha: The weight of classification loss.
        # >> - dim_in: The dimension of the input sample.
        # >> - num_classes: The number of classes.
        # >> - dim_z: The dimension of the hidden variable z.
        # >> - dim_hidden_de: The hidden layer dimension of the decoder.
        # >> - dim_hidden_en_y: The hidden layer dimension of the encoder for y.
        # >> - dim_hidden_en_z: The hidden layer dimension of the encoder for z.
        # >> - activations_de: The activation functions of the decoder.
        # >> - activations_en_y: The activation functions of the encoder for y.
        # >> - activations_en_z: The activation functions of the encoder for z.
        # >> - num_labeled: The number of labeled samples.
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
                                    verbose=verbose
                                    )
        self.dim_z=dim_z
        self.dim_in=dim_in
        self.num_classes=num_classes
        self.alpha=alpha
        self.dim_hidden_de = dim_hidden_de
        self.dim_hidden_en_y = dim_hidden_en_y
        self.dim_hidden_en_z = dim_hidden_en_z
        self.activations_de = activations_de
        self.activations_en_y = activations_en_y
        self.activations_en_z = activations_en_z
        self.num_labeled=num_labeled
        self._estimator_type = ClassifierMixin._estimator_type

    def start_fit(self):
        self.init_epoch()
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_classes
        self.num_labeled = self.num_labeled if self.num_labeled is not None else \
            self._train_dataset.labeled_dataset.X.shape[0]
        self.dim_in=self.dim_in if self.dim_in is not None else \
            self._train_dataset.labeled_dataset.X.shape[1:]
        if self.network is None:
            self.network=VAE.SSVAE(dim_in=self.dim_in,num_classes=self.num_classes,dim_z=self.dim_z,
                               dim_hidden_de=self.dim_hidden_de,activations_de=self.activations_de,
                               dim_hidden_en_y=self.dim_hidden_en_y, activations_en_y=self.activations_en_y,
                               dim_hidden_en_z=self.dim_hidden_en_z, activations_en_z=self.activations_en_z,
                               device=self.device)
            self._network=copy.deepcopy(self.network)
            self.init_model()
            self.init_ema()
            self.init_optimizer()
            self.init_scheduler()
        self._network.zero_grad()
        self._network.train()

    def loss_components_fn(self,x, y, z, p_y, p_z, p_x_yz, q_z_xy):
        return - p_x_yz.log_prob(x).sum(1) \
               - p_y.log_prob(y) \
               - p_z.log_prob(z).sum(1) \
               + q_z_xy.log_prob(z).sum(1)

    def train(self,lb_X=None,lb_y=None,ulb_X=None,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X = lb_X[0] if isinstance(lb_X,(list,tuple)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(list,tuple)) else lb_y
        ulb_X=ulb_X[0]if isinstance(ulb_X,(list,tuple)) else ulb_X
        lb_X=lb_X.view(lb_X.shape[0],-1).bernoulli()
        ulb_X = ulb_X.view(ulb_X.shape[0], -1).bernoulli()
        lb_q_y = self._network.encode_y(lb_X)
        ulb_q_y = self._network.encode_y(ulb_X)

        lb_y = one_hot(lb_y, self.num_classes,self.device).to(self.device)

        lb_q_z_xy = self._network.encode_z(lb_X, lb_y)

        lb_z = lb_q_z_xy.rsample()
        lb_p_x_yz = self._network.decode(lb_y, lb_z)

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
        sup_loss=sup_loss.mean(0)
        probs=F.softmax(lb_q_y.probs,dim=-1)
        cls_loss=- self.alpha  * lb_q_y.log_prob(lb_y)
        cls_loss=cls_loss.mean(0)


        unsup_loss = - ulb_q_y.entropy()
        idx=0
        for ulb_y in ulb_q_y.enumerate_support():
            L_xy = self.loss_components_fn(ulb_X, ulb_y, ulb_z_list[idx], ulb_p_y, ulb_p_z, ulb_p_x_yz_list[idx], ulb_q_z_xy_list[idx])
            unsup_loss += ulb_q_y.log_prob(ulb_y).exp() * L_xy
            idx+=1
        unsup_loss=unsup_loss.mean(0)
        loss=sup_loss+unsup_loss+cls_loss
        return loss

    def optimize(self,loss,*args,**kwargs):
        self._network.zero_grad()
        loss.backward()
        self._optimizer.step()

    @torch.no_grad()
    def estimate(self, X, idx=None, *args, **kwargs):
        X=X.view(X.shape[0],-1).bernoulli()
        outputs = self._network(X)
        return outputs

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)

    def generate(self,num,z=None,x=None,y=None):
        if y is not None:
            y = one_hot(y,self.num_classes,self.device).to(self.device)
        if x is not None and y is None:
            y=self._network.encode_y(x)
        if x is not None and y is not None and z is None:
            z = self._network.encode_z(x,y)
        z = Variable(torch.randn(num, self.dim_z).to(self.device)) if z is None else z
        y = one_hot(Variable(torch.LongTensor([random.randrange(self.num_classes) for _ in range(num)]).to(self.device)),nClass=self.num_classes,device=self.device)
        result=self._network.decode(z,y).probs
        result = result.view(tuple([result.shape[0]]) + tuple(self.dim_in)).detach().numpy()
        return result
