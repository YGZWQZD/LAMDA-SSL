from math import ceil

from Semi_sklearn.Dataset.LabledDataset import LabledDataset
from Semi_sklearn.Dataset.SemiTrainDataset import SemiTrainDataset
from Semi_sklearn.Data_loader.SemiTrainDataloader import SemiTrainDataLoader
from Semi_sklearn.Data_loader.SemiTestDataloader import SemiTestDataLoader
from abc import abstractmethod
class SemiDeepModelMixin:
    def __init__(self, train_dataset=None, test_dataset=None,
                 train_dataloader=None,
                 test_dataloader=None,
                 augmentation=None,
                 network=None,
                 epoch=1,
                 it_epoch=None,
                 it_total=None,
                 mu=None,# unlabled/labled in each batch
                 optimizer=None,
                 scheduler=None,
                 lr=0.01,
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
        self.lr=lr
        self.device=device
        self.y_est=None
        if it_epoch is not None:
            num_it_total=epoch*it_epoch
        if it_total is not None:
            num_it_epoch=ceil(it_total/epoch)

    def fit(self,labled_X=None,labled_y=None,unlabled_X=None,labled_dataset=None,unlabled_dataset=None,train_dataset=None):
        if train_dataset is not None:
            self.train_dataset=train_dataset
        else:
            if labled_X is not None:
                self.train_dataset.init_dataset(labled_X=labled_X,labled_y=labled_y,unlabled_X=unlabled_X,mu=self.mu)
            elif labled_dataset is not None:
                self.train_dataset.init_dataset(labled_dataset=labled_dataset,unlabled_dataset=unlabled_dataset,mu=self.mu)
        self.labled_dataloader,self.unlabled_dataloader=self.train_dataloader.get_dataloader(self.train_dataset)
        self.start_fit()
        self.it_total=0
        for _ in range(self.epoch):
            self.it_epoch=0
            self.start_epoch()
            for (lb_X, lb_y), (ulb_X, _) in zip(self.labled_dataloader,self.unlabled_dataloader):
                self.it_total+=1
                self.it_epoch+=1
                self.start_batch_train()
                train_result=self.train(lb_X,lb_y,ulb_X)
                loss=self.get_loss(train_result)
                self.backward(loss)
                self.end_batch_train()
            self.end_epoch()
        self.end_fit()
    def predict(self,test_X=None,test_dataset=None):
        if test_dataset is not None:
            self.test_dataset=test_dataset
        else:
            self.test_dataset.init_dataset(X=test_X)
        self.test_dataloader=self.test_dataloader.get_dataloader(self.test_dataset)
        self.y_est=[]
        self.start_predict()
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







