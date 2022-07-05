from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
import torch
import numpy as np
from LAMDA_SSL.utils import class_status
from LAMDA_SSL.Transform.Rotate import Rotate
from LAMDA_SSL.utils import Bn_Controller
import LAMDA_SSL.Config.S4L as config
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
from LAMDA_SSL.Loss.Semi_supervised_Loss import Semi_supervised_loss


class S4L(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 lambda_u=config.lambda_u,
                 num_classes=config.num_classes,
                 p_target=config.p_target,
                 rotate_v_list=config.rotate_v_list,
                 labeled_usp=config.labeled_usp,
                 all_rot=config.all_rot,
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
                 verbose=config.verbose
                 ):
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
        self.num_classes=num_classes
        self.rotate_v_list=rotate_v_list
        self.p_model = None
        self.p_target=p_target
        self.labeled_usp=labeled_usp
        self.all_rot=all_rot
        self.bn_controller = Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_transform(self.weakly_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weakly_augmentation,dim=1,x=0,y=0)

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_classes
        self._network.zero_grad()
        self._network.train()

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):

        lb_x_w=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y
        ulb_x_w = ulb_X[0] if isinstance(ulb_X, (tuple, list)) else ulb_X

        logits_x_lb_w = self._network(lb_x_w)[0]

        rot_x = torch.Tensor().to(self.device)
        rot_y = []

        for item in ulb_x_w:
            if self.all_rot:
                for _v in self.rotate_v_list:
                    rot_x = torch.cat((rot_x, Rotate(v=_v).fit_transform(item).unsqueeze(0)), dim=0)
                    rot_y.append(self.rotate_v_list.index(_v))
            else:
                _v = np.random.choice(self.rotate_v_list, 1).item()
                rot_x = torch.cat((rot_x, Rotate(v=_v).fit_transform(item).unsqueeze(0)), dim=0)
                rot_y.append(self.rotate_v_list.index(_v))
        if self.labeled_usp:
            for item in lb_x_w:
                if self.all_rot:
                    for _v in self.rotate_v_list:
                        rot_x = torch.cat((rot_x, Rotate(v=_v).fit_transform(item).unsqueeze(0)), dim=0)
                        rot_y.append(self.rotate_v_list.index(_v))
                else:
                    _v = np.random.choice(self.rotate_v_list, 1).item()
                    rot_x = torch.cat((rot_x, Rotate(v=_v).fit_transform(item).unsqueeze(0)), dim=0)
                    rot_y.append(self.rotate_v_list.index(_v))

        rot_y = torch.LongTensor(rot_y).to(self.device)

        logits_x_rot = self._network(rot_x)[1]

        return logits_x_lb_w,lb_y,logits_x_rot,rot_y


    def get_loss(self,train_result,*args,**kwargs):
        logits_x_lb_w,lb_y,logits_x_rot,rot_y=train_result
        sup_loss = Cross_Entropy(reduction='mean')(logits_x_lb_w, lb_y)  # CE_loss for labeled data
        rot_loss = Cross_Entropy(reduction='mean')(logits_x_rot, rot_y)
        loss = Semi_supervised_loss(self.lambda_u)(sup_loss,rot_loss)
        return loss

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)



