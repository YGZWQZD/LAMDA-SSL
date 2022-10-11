import copy
from math import ceil
import torch
from LAMDA_SSL.Base.SemiEstimator import SemiEstimator
from torch.utils.data.dataset import Dataset
from LAMDA_SSL.Dataset.TrainDataset import TrainDataset

from LAMDA_SSL.Base.BaseOptimizer import BaseOptimizer
from LAMDA_SSL.Base.BaseScheduler import BaseScheduler

from LAMDA_SSL.utils import EMA
from LAMDA_SSL.utils import to_device
from torch.nn import Softmax
from LAMDA_SSL.Dataloader.TrainDataloader import TrainDataLoader

class DeepModelMixin(SemiEstimator):
    def __init__(self, train_dataset=None,
                 labeled_dataset=None,
                 unlabeled_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 train_dataloader=None,
                 labeled_dataloader=None,
                 unlabeled_dataloader=None,
                 valid_dataloader=None,
                 test_dataloader=None,
                 augmentation=None,
                 network=None,
                 epoch=1,
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
                 parallel=None,
                 file=None,
                 verbose=True
                 ):
        # >> Parameter
        # >> - train_dataset: Data manager for training data.
        # >> - labeled_dataset: Data manager for labeled data.
        # >> - unlabeled_dataset: Data manager for unlabeled data.
        # >> - valid_dataset: Data manager for valid data.
        # >> - test_dataset: Data manager for test data.
        # >> - augmentation: Augmentation method, if there are multiple augmentation methods, you can use a dictionary or a list to pass parameters.
        # >> - network: The backbone neural network.
        # >> - epoch: Number of training epochs.
        # >> - num_it_epoch: The number of iterations in each round, that is, the number of batches of data.
        # >> - num_it_total: The total number of batches.
        # >> - eval_epoch: Model evaluation is performed every eval_epoch epochs.
        # >> - eval_it: Model evaluation is performed every eval_it iterations.
        # >> - mu: The ratio of the number of unlabeled data to the number of labeled data.
        # >> - optimizer: The optimizer used in training.
        # >> - weight_decay: The optimizer's learning rate decay parameter.
        # >> - ema_decay: The update scale for the exponential moving average of the model parameters.
        # >> - scheduler: Learning rate scheduler.
        # >> - device: Training equipment.
        # >> - evaluation: Model evaluation metrics. If there are multiple metrics, a dictionary or a list can be used.
        # >> - train_sampler: Sampler of training data.
        # >> - labeled_sampler=None: Sampler of labeled data.
        # >> - unlabeled_sampler=None: Sampler of unlabeled data.
        # >> - train_batch_sampler=None: Batch sampler of training data
        # >> - labeled_batch_sampler: Batch sampler of labeled data
        # >> - unlabeled_batch_sampler: Batch sampler of unlabeled data
        # >> - valid_sampler: sampler of valid data.
        # >> - valid_batch_sampler: Batch sampler of valid data.
        # >> - test_sampler: Sampler of test data.
        # >> - test_batch_sampler: Batch sampler of test data.
        # >> - parallel: Distributed training method.
        # >> - file: Output file.
        self.train_dataset=train_dataset if train_dataset is not None else TrainDataset(labeled_dataset=labeled_dataset,
                                                                                        unlabeled_dataset=unlabeled_dataset)
        self.labeled_dataset=labeled_dataset
        self.unlabeled_dataset=unlabeled_dataset
        self.valid_dataset = valid_dataset if valid_dataset is not None else test_dataset
        self.test_dataset=test_dataset
        self.train_dataloader=train_dataloader
        self.labeled_dataloader=labeled_dataloader
        self.unlabeled_dataloader=unlabeled_dataloader
        self.valid_dataloader=valid_dataloader if valid_dataloader is not None else test_dataloader
        self.test_dataloader=test_dataloader
        self.augmentation=augmentation
        self.network=network
        self.epoch=epoch
        self.mu=mu
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.device=device
        self.eval_epoch=eval_epoch
        self.eval_it=eval_it
        self.weight_decay=weight_decay
        self.ema_decay=ema_decay
        self.y_est=None
        self.y_true=None
        self.y_pred=None
        self.y_score=None
        self.num_it_epoch=num_it_epoch
        self.num_it_total=num_it_total
        self.evaluation=evaluation

        self.train_sampler=train_sampler
        self.train_batch_sampler=train_batch_sampler

        self.labeled_sampler=labeled_sampler
        self.unlabeled_sampler=unlabeled_sampler
        self.labeled_batch_sampler=labeled_batch_sampler
        self.unlabeled_batch_sampler = unlabeled_batch_sampler

        self.valid_sampler=valid_sampler if valid_sampler is not None else test_sampler
        self.valid_batch_sampler=valid_batch_sampler if valid_batch_sampler is not None else test_batch_sampler

        self.test_sampler=test_sampler
        self.test_batch_sampler=test_batch_sampler

        self.valid_performance=None

        self.parallel=parallel
        self.verbose=verbose
        self.it_epoch=0
        self.it_total=0
        self.loss=None
        self.weak_augmentation=None
        self.strong_augmentation=None
        self.normalization=None
        self.performance=None
        self.valid_performance=None
        self.ema=None
        if isinstance(file,str):
            file=open(file,"w")
        self.file=file
        self._estimator_type=None

    def init_model(self):
        self._network = copy.deepcopy(self.network)
        self._parallel = copy.deepcopy(self.parallel)
        if self.device is None:
            self.device='cpu'
        if self.device is not 'cpu':
            torch.cuda.set_device(self.device)
        self._network=self._network.to(self.device)
        if self._parallel is not None:
            self._network=self._parallel.init_parallel(self._network)

    def init_ema(self):
        if self.ema_decay is not None:
            self.ema=EMA(model=self._network,decay=self.ema_decay)
            self.ema.register()
        else:
            self.ema=None

    def init_optimizer(self):
        self._optimizer=copy.deepcopy(self.optimizer)
        if isinstance(self._optimizer,BaseOptimizer):
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in self._network.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self._network.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self._optimizer=self._optimizer.init_optimizer(params=grouped_parameters)

    def init_scheduler(self):
        self._scheduler=copy.deepcopy(self.scheduler)
        if isinstance(self._scheduler,BaseScheduler):
            self._scheduler=self._scheduler.init_scheduler(optimizer=self._optimizer)

    def init_epoch(self):
        if self.num_it_epoch is not None and self.epoch is not None:
            self.num_it_total=self.epoch*self.num_it_epoch
        elif self.num_it_total is not None and self.epoch is not None:
            self.num_it_epoch=ceil(self.num_it_total/self.epoch)
        elif self.num_it_total is not None and self.num_it_epoch is not None:
            self.epoch=ceil(self.num_it_total/self.num_it_epoch)

    def init_augmentation(self):
        self._augmentation = copy.deepcopy(self.augmentation)
        if self._augmentation is not None:
            if isinstance(self._augmentation, dict):
                self.weak_augmentation = self._augmentation['augmentation'] \
                    if 'augmentation' in self._augmentation.keys() \
                    else self._augmentation['weak_augmentation']
                if 'strong_augmentation' in self._augmentation.keys():
                    self.strong_augmentation = self._augmentation['strong_augmentation']
            elif isinstance(self._augmentation, (list, tuple)):
                self.weak_augmentation = self._augmentation[0]
                if len(self._augmentation) > 1:
                    self.strong_augmentation = self._augmentation[1]
            else:
                self.weak_augmentation = copy.copy(self._augmentation)
            if self.strong_augmentation is None:
                self.strong_augmentation = copy.copy(self.weak_augmentation)

    def init_transform(self):
        if self.weak_augmentation is not None:
            self._train_dataset.add_transform(self.weak_augmentation,dim=1,x=0,y=0)
            self._train_dataset.add_unlabeled_transform(self.weak_augmentation, dim=1, x=0, y=0)

    def init_train_dataset(self,X=None,y=None,unlabeled_X=None, *args, **kwargs):
        self._train_dataset=copy.deepcopy(self.train_dataset)
        if isinstance(X,TrainDataset):
            self._train_dataset=X
        elif isinstance(X,Dataset) and y is None:
            self._train_dataset.init_dataset(labeled_dataset=X, unlabeled_dataset=unlabeled_X)
        else:
            self._train_dataset.init_dataset(labeled_X=X, labeled_y=y,unlabeled_X=unlabeled_X)

    def init_train_dataloader(self):
        self._train_dataloader=copy.deepcopy(self.train_dataloader)
        self._labeled_dataloader = copy.deepcopy(self.labeled_dataloader)
        self._unlabeled_dataloader = copy.deepcopy(self.unlabeled_dataloader)
        self._train_sampler=copy.deepcopy(self.train_sampler)
        self._labeled_sampler = copy.deepcopy(self.labeled_sampler)
        self._unlabeled_sampler = copy.deepcopy(self.unlabeled_sampler)
        self._train_batch_sampler=copy.deepcopy(self.train_batch_sampler)
        self._labeled_batch_sampler = copy.deepcopy(self.labeled_batch_sampler)
        self._unlabeled_batch_sampler = copy.deepcopy(self.unlabeled_batch_sampler)
        if self._train_dataloader is not None:
            self._labeled_dataloader,self._unlabeled_dataloader=self._train_dataloader.init_dataloader(dataset=self._train_dataset,
                                                                                       sampler=self._train_sampler,
                                                                                       batch_sampler=self._train_batch_sampler,
                                                                                       mu=self.mu)
        else:
            self._train_dataloader=TrainDataLoader(labeled_dataloader=self._labeled_dataloader,unlabeled_dataloader=self._unlabeled_dataloader)
            self._train_sampler={'labeled':self._labeled_sampler,'unlabeled':self._unlabeled_sampler}
            self._train_batch_sampler={'labeled':self._labeled_batch_sampler,'unlabeled':self._unlabeled_batch_sampler}
            self._labeled_dataloader, self._unlabeled_dataloader = self._train_dataloader.init_dataloader(
                dataset=self._train_dataset,
                sampler=self._train_sampler,
                batch_sampler=self._train_batch_sampler,
                mu=self.mu)

    def start_fit(self, *args, **kwargs):
        self.init_epoch()
        self._network.zero_grad()
        self._network.train()

    def start_fit_epoch(self, *args, **kwargs):
        pass

    def start_fit_batch(self, *args, **kwargs):
        pass

    def train(self,lb_X=None,lb_y=None,ulb_X=None,lb_idx=None,ulb_idx=None,*args,**kwargs):
        raise NotImplementedError

    def get_loss(self,train_result,*args,**kwargs):
        raise NotImplementedError

    def optimize(self,loss,*args,**kwargs):
        self._network.zero_grad()
        loss.backward()
        self._optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()
        if self.ema is not None:
            self.ema.update()

    def end_fit_batch(self, train_result,*args, **kwargs):
        self.loss = self.get_loss(train_result)
        self.optimize(self.loss)

    def fit_batch_loop(self,valid_X=None,valid_y=None):
        for (lb_idx, lb_X, lb_y), (ulb_idx, ulb_X, _) in zip(self._labeled_dataloader, self._unlabeled_dataloader):
            if self.it_epoch >= self.num_it_epoch or self.it_total >= self.num_it_total:
                break
            self.start_fit_batch()
            lb_idx = to_device(lb_idx,self.device)
            lb_X = to_device(lb_X,self.device)
            lb_y = to_device(lb_y,self.device)
            ulb_idx = to_device(ulb_idx,self.device)
            ulb_X  = to_device(ulb_X,self.device)
            train_result = self.train(lb_X=lb_X, lb_y=lb_y, ulb_X=ulb_X, lb_idx=lb_idx, ulb_idx=ulb_idx)
            self.end_fit_batch(train_result)
            self.it_total += 1
            self.it_epoch += 1
            if self.verbose:
                print(self.it_total,file=self.file)
                print(self.it_total)
            if valid_X is not None and self.eval_it is not None and self.it_total % self.eval_it == 0:
                self.evaluate(X=valid_X, y=valid_y,valid=True)
                self.valid_performance.update({"epoch_" + str(self._epoch) + "_it_" + str(self.it_epoch): self.performance})

    def end_fit_epoch(self, *args, **kwargs):
        pass

    def fit_epoch_loop(self,valid_X=None,valid_y=None):
        self.valid_performance={}
        self.it_total = 0
        for self._epoch in range(1,self.epoch+1):
            self.it_epoch=0
            if self.it_total >= self.num_it_total:
                break
            self.start_fit_epoch()
            self.fit_batch_loop(valid_X,valid_y)
            self.end_fit_epoch()
            if valid_X is not None and self.eval_epoch is not None and self._epoch % self.eval_epoch==0:
                self.evaluate(X=valid_X,y=valid_y,valid=True)
                self.valid_performance.update({"epoch_" + str(self._epoch) + "_it_" + str(self.it_epoch): self.performance})

        if valid_X is not None and (self.eval_epoch is None or self.epoch% self.eval_epoch!=0):
            self.evaluate(X=valid_X, y=valid_y, valid=True)
            self.valid_performance.update({"epoch_" + str(self._epoch) + "_it_" + str(self.it_epoch): self.performance})

    def end_fit(self, *args, **kwargs):
        pass

    def fit(self,X=None,y=None,unlabeled_X=None,valid_X=None,valid_y=None):
        self.init_train_dataset(X,y,unlabeled_X)
        self.init_train_dataloader()
        if self.network is not None:
            self.init_model()
            self.init_ema()
            self.init_optimizer()
            self.init_scheduler()
        self.init_augmentation()
        self.init_transform()
        self.start_fit()
        self.fit_epoch_loop(valid_X,valid_y)
        self.end_fit()
        return self

    def init_estimate_dataset(self, X=None,valid=False):
        self._valid_dataset = copy.deepcopy(self.valid_dataset)
        self._test_dataset=copy.deepcopy(self.test_dataset)
        if valid:
            if isinstance(X,Dataset):
                self._valid_dataset=X
            else:
                self._valid_dataset=self._valid_dataset.init_dataset(X=X)
        else:
            if isinstance(X,Dataset):
                self._test_dataset=X
            else:
                self._test_dataset=self._test_dataset.init_dataset(X=X)

    def init_estimate_dataloader(self,valid=False):
        self._valid_dataloader=copy.deepcopy(self.valid_dataloader)
        self._test_dataloader = copy.deepcopy(self.test_dataloader)
        self._valid_sampler = copy.deepcopy(self.valid_sampler)
        self._test_sampler=copy.deepcopy(self.test_sampler)
        self._valid_batch_sampler=copy.deepcopy(self.valid_batch_sampler)
        self._test_batch_sampler=copy.deepcopy(self.test_batch_sampler)
        if valid:
            self._estimate_dataloader=self._valid_dataloader.init_dataloader(self._valid_dataset,
                                                            sampler=self._valid_sampler,
                                                            batch_sampler=self._valid_batch_sampler)
        else:
            self._estimate_dataloader=self._test_dataloader.init_dataloader(self._test_dataset,
                                                            sampler=self._test_sampler,
                                                            batch_sampler=self._test_batch_sampler)

    def start_predict(self, *args, **kwargs):
        self._network.eval()
        if self.ema is not None:
            self.ema.apply_shadow()
        self.y_est = torch.Tensor().to(self.device)

    def start_predict_batch(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def estimate(self, X, idx=None, *args, **kwargs):
        outputs = self._network(X)
        return outputs

    def end_predict_batch(self, *args, **kwargs):
        pass

    def predict_batch_loop(self):
        with torch.no_grad():
            for idx,X,_ in self._estimate_dataloader:
                self.start_predict_batch()
                idx=to_device(idx,self.device)
                X=X[0] if isinstance(X,(list,tuple)) else X
                X=to_device(X,self.device)
                _est = self.estimate(X=X,idx=idx)
                _est = _est[0] if  isinstance(_est,(list,tuple)) else _est
                self.y_est=torch.cat((self.y_est,_est),0)
                self.end_predict_batch()

    @torch.no_grad()
    def get_predict_result(self, y_est, *args, **kwargs):
        if self._estimator_type == 'classifier' or 'classifier' in self._estimator_type:
            y_score = Softmax(dim=-1)(y_est)
            max_probs, y_pred = torch.max(y_score, dim=-1)
            y_pred = y_pred.cpu().detach().numpy()
            self.y_score = y_score.cpu().detach().numpy()
            return y_pred
        else:
            self.y_score = y_est.cpu().detach().numpy()
            y_pred = self.y_score
            return y_pred

    def end_predict(self, *args, **kwargs):
        self.y_pred = self.get_predict_result(self.y_est)
        if self.ema is not None:
            self.ema.restore()
        self._network.train()

    @torch.no_grad()
    def predict(self,X=None,valid=False):
        self.init_estimate_dataset(X,valid)
        self.init_estimate_dataloader(valid)
        self.start_predict()
        self.predict_batch_loop()
        self.end_predict()
        return self.y_pred

    @torch.no_grad()
    def predict_proba(self,X=None,valid=False):
        self.init_estimate_dataset(X,valid)
        self.init_estimate_dataloader(valid)
        self.start_predict()
        self.predict_batch_loop()
        self.end_predict()
        return self.y_score

    @torch.no_grad()
    def evaluate(self,X,y=None,valid=False):

        if isinstance(X,Dataset) and y is None:
            y=getattr(X,'y')

        self.y_pred=self.predict(X,valid=valid)
        self.y_score=self.y_score

        if self.evaluation is None:
            return None
        elif isinstance(self.evaluation,(list,tuple)):
            performance=[]
            for eval in self.evaluation:
                if self._estimator_type == 'classifier' or 'classifier' in self._estimator_type:
                    score=eval.scoring(y,self.y_pred,self.y_score)
                else:
                    score = eval.scoring(y,self.y_pred)
                performance.append(score)
                if self.verbose:
                    print(score, file=self.file)
                    print(score)
            self.performance = performance
            return performance
        elif isinstance(self.evaluation,dict):
            performance={}
            for key,val in self.evaluation.items():
                if self._estimator_type == 'classifier' or 'classifier' in self._estimator_type:
                    performance[key]=val.scoring(y,self.y_pred,self.y_score)
                else:
                    performance[key] = val.scoring(y, self.y_pred)
                if self.verbose:
                    print(key,' ',performance[key],file=self.file)
                    print(key, ' ', performance[key])
                self.performance = performance
            return performance
        else:
            if self._estimator_type == 'classifier' or 'classifier' in self._estimator_type:
                performance=self.evaluation.scoring(y,self.y_pred,self.y_score)
            else:
                performance = self.evaluation.scoring(y, self.y_pred)
            if self.verbose:
                print(performance, file=self.file)
                print(performance)
            self.performance=performance
            return performance