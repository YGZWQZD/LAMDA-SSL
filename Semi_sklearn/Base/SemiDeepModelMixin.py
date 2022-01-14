from math import ceil
import torch
from Semi_sklearn.Base.SemiEstimator import SemiEstimator
from Semi_sklearn.Dataset.LabledDataset import LabledDataset
from Semi_sklearn.Dataset.SemiTrainDataset import SemiTrainDataset
from Semi_sklearn.Data_loader.SemiTrainDataloader import SemiTrainDataLoader
from Semi_sklearn.Data_loader.SemiTestDataloader import SemiTestDataLoader
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
                 mu=None,# unlabled/labled in each batch
                 optimizer=None,
                 scheduler=None,
                 device=None
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
        self.y_est=None
        self.num_it_epoch=num_it_epoch
        self.num_it_total=num_it_total
        if self.num_it_epoch is not None and self.epoch is not None:
            self.num_it_total=self.epoch*self.num_it_epoch
        if self.num_it_total is not None and self.epoch is not None:
            self.num_it_epoch=ceil(self.num_it_total/self.epoch)
        if self.num_it_total is not None and self.num_it_epoch is not None:
            self.epoch=ceil(self.num_it_total/self.num_it_epoch)

    def fit(self,labled_X=None,labled_y=None,unlabled_X=None,labled_dataset=None,unlabled_dataset=None,train_dataset=None):
        if train_dataset is not None:
            self.train_dataset=train_dataset
        else:
            if labled_X is not None:
                self.train_dataset.init_dataset(labled_X=labled_X,labled_y=labled_y,unlabled_X=unlabled_X)
            elif labled_dataset is not None:
                self.train_dataset.init_dataset(labled_dataset=labled_dataset,unlabled_dataset=unlabled_dataset)
        self.labled_dataloader,self.unlabled_dataloader=self.train_dataloader.get_dataloader(dataset=self.train_dataset,mu=self.mu)
        print(self.labled_dataloader)
        print(self.unlabled_dataloader)
        self.start_fit()
        self.it_total=0
        for _ in range(self.epoch):
            self.it_epoch=0
            self.start_epoch()
            for (lb_X, lb_y), (ulb_X, _) in zip(self.labled_dataloader,self.unlabled_dataloader):
                self.it_total+=1
                self.it_epoch+=1
                print(self.it_total)
                print(lb_X.shape)
                print(ulb_X.shape)
                if self.it_epoch >= self.num_it_epoch or self.it_total>=self.num_it_total:
                    break
                self.start_batch_train()
                train_result=self.train(lb_X,lb_y,ulb_X)
                loss=self.get_loss(train_result)
                print(loss)
                self.backward(loss)
                self.end_batch_train()
            self.end_epoch()
            if self.it_total>=self.num_it_total:
                break
        self.end_fit()
        return self

    def predict(self,test_X=None,test_dataset=None):
        if test_dataset is not None:
            self.test_dataset=test_dataset
        else:
            self.test_dataset.init_dataset(X=test_X)
        self.test_dataloader=self.test_dataloader.get_dataloader(self.test_dataset)
        self.y_est=[]
        self.start_predict()
        with torch.no_grad():
            for X,_ in self.test_dataloader:
                self.start_batch_test()
                self.y_est.append(self.estimate(X))
                self.end_batch_test()
            y_pred=self.get_predict_result(self.y_est)
            self.end_predict()
        return y_pred

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
    def backward(self,loss,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def estimate(self,X,*args,**kwargs):
        raise NotImplementedError
    @abstractmethod
    def get_predict_result(self,y_est,*args,**kwargs):
        raise NotImplementedError







