import copy
from math import ceil
import torch
from Semi_sklearn.Base.SemiEstimator import SemiEstimator
from torch.utils.data.dataset import Dataset
from Semi_sklearn.Dataset.SemiTrainDataset import SemiTrainDataset
from abc import abstractmethod
class SemiDeepModelMixin(SemiEstimator):
    def __init__(self, train_dataset=None, test_dataset=None,
                 train_dataloader=None,
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
                 scheduler=None,
                 device=None,
                 evaluation=None,
                 train_sampler=None,
                 train_batch_sampler=None,
                 test_sampler=None,
                 test_batch_Sampler=None,
                 parallel=None
                 ):
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        self.train_dataloader=train_dataloader
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
        self.y_est=None
        self.y_true=None
        self.y_pred=None
        self.y_score=None
        self.num_it_epoch=num_it_epoch
        self.num_it_total=num_it_total
        self.evaluation=evaluation
        self.train_sampler=train_sampler
        self.train_batch_sampler=train_batch_sampler
        self.test_sampler=test_sampler
        self.test_batch_sampler=test_batch_Sampler
        self.parallel=parallel
        self._optimizer=copy.deepcopy(self.optimizer)
        self._network=copy.deepcopy(self.network)
        self._scheduler=copy.deepcopy(self.scheduler)
        self._train_sampler=copy.deepcopy(self.train_sampler)
        self._test_sampler=copy.deepcopy(self.test_sampler)
        self._train_batch_sampler=copy.deepcopy(self.train_batch_sampler)
        self._test_batch_sampler=copy.deepcopy(self.test_batch_sampler)
        self._train_dataset=copy.deepcopy(self.train_dataset)
        self._test_dataset=copy.deepcopy(self.test_dataset)
        self._train_dataloader=copy.deepcopy(self.train_dataloader)
        self._test_dataloader = copy.deepcopy(self.test_dataloader)
        self._augmentation=copy.deepcopy(self.augmentation)
        self._evaluation=copy.deepcopy(self.evaluation)
        self._parallel=copy.deepcopy(self.parallel)
        self.init_model()


    def fit(self,X=None,y=None,unlabled_X=None,valid_X=None,valid_y=None):

        if self.num_it_epoch is not None and self.epoch is not None:
            self.num_it_total=self.epoch*self.num_it_epoch
        if self.num_it_total is not None and self.epoch is not None:
            self.num_it_epoch=ceil(self.num_it_total/self.epoch)
        if self.num_it_total is not None and self.num_it_epoch is not None:
            self.epoch=ceil(self.num_it_total/self.num_it_epoch)

        if isinstance(X,SemiTrainDataset):
            self._train_dataset=X
        elif isinstance(X,Dataset) and y is None:
            self._train_dataset.init_dataset(labled_dataset=X, unlabled_dataset=unlabled_X)
        else:
            self._train_dataset.init_dataset(labled_X=X, labled_y=y,unlabled_X=unlabled_X)
        labled_dataloader,unlabled_dataloader=self._train_dataloader.get_dataloader(dataset=self._train_dataset,
                                                                                   sampler=self._train_sampler,
                                                                                   batch_sampler=self._train_batch_sampler,
                                                                                   mu=self.mu)
        self.start_fit()
        self.it_total=0
        self._network.zero_grad()
        self._network.train()
        for _epoch in range(self.epoch):
            self.it_epoch=0
            if self.it_total>=self.num_it_total:
                break
            self.start_epoch()
            for (lb_X, lb_y), (ulb_X, _) in zip(labled_dataloader,unlabled_dataloader):
                if self.it_epoch >= self.num_it_epoch or self.it_total>=self.num_it_total:
                    break
                self.start_batch_train()
                train_result=self.train(lb_X,lb_y,ulb_X)
                loss=self.get_loss(train_result)
                loss.backward()
                self.optimize()
                self.end_batch_train()
                self.it_total+=1
                self.it_epoch+=1
                print(self.it_total)
                if valid_X is not None and self.eval_it is not None and self.it_total % self.eval_it == 0:
                    # print('valid')
                    # print(valid_X)
                    self.evaluate(X=valid_X, y=valid_y)
            self.end_epoch()
            if valid_X is not None and self.eval_epoch is not None and _epoch % self.eval_epoch==0:
                # print('valid')
                self.evaluate(X=valid_X,y=valid_y)
        self.end_fit()
        return self

    def predict(self,X=None):
        if isinstance(X,Dataset):
            self._test_dataset=X
        else:
            self._test_dataset.init_dataset(X=X)
        # print(self._test_dataset)
        test_dataloader=self._test_dataloader.get_dataloader(self._test_dataset,
                                                            sampler=self._test_sampler,
                                                            batch_sampler=self._test_batch_sampler)
        self.y_est=torch.Tensor([])
        self.start_predict()
        self._network.eval()
        with torch.no_grad():
            for X,_ in test_dataloader:
                self.start_batch_test()
                self.y_est=torch.cat((self.y_est,self.estimate(X)),0)
                self.end_batch_test()
            self.y_pred=self.get_predict_result(self.y_est)
            self.end_predict()
        self._network.train()
        print(self._test_dataset)
        return self.y_pred

    def evaluate(self,X,y=None):

        if isinstance(X,Dataset) and y is None:
            y=getattr(X,'y')

        y_pred=self.predict(X).cpu()
        y_score=self.y_score.cpu()
        if self.evaluation is None:
            return None
        elif isinstance(self.evaluation,(list,tuple)):
            result=[]
            for eval in self.evaluation:
                result.append(eval.scoring(y,y_pred,y_score))
            return result
        elif isinstance(self.evaluation,dict):
            result={}
            for key,val in self.evaluation.items():
                result[key]=val.scoring(y,y_pred,y_score)
                print(key,' ',result[key])
            return result
        else:
            result=self.evaluation.scoring(y,y_pred,y_score)
            return result

    def init_model(self):
        if self.device is not None and self.device is not 'cpu':
            torch.cuda.set_device(self.device)
        self._network=self._network.to(self.device)
        if self._parallel is not None:
            self._parallel=self._parallel.init_parallel(self._network)

    def start_fit(self, *args, **kwargs):
        pass
    def start_epoch(self, *args, **kwargs):
        pass
    def start_batch_train(self, *args, **kwargs):
        pass
    def end_batch_train(self, *args, **kwargs):
        pass
    def start_batch_test(self, *args, **kwargs):
        pass
    def end_batch_test(self, *args, **kwargs):
        pass
    def end_fit(self, *args, **kwargs):
        pass
    def end_epoch(self, *args, **kwargs):
        pass
    def end_batch(self, *args, **kwargs):
        pass
    def start_predict(self, *args, **kwargs):
        pass
    def end_predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self,lb_X,lb_y,ulb_X,*args,**kwargs):
        raise NotImplementedError
    @abstractmethod
    def get_loss(self,train_result,*args,**kwargs):
        raise NotImplementedError
    @abstractmethod
    def optimize(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def estimate(self,X,*args,**kwargs):
        raise NotImplementedError
    @abstractmethod
    def get_predict_result(self,y_est,*args,**kwargs):
        raise NotImplementedError









