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
                 mu=None,# unlabled/labled in each batch
                 optimizer=None,
                 scheduler=None,
                 device=None,
                 evaluation=None
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


    def fit(self,X=None,y=None,unlabled_X=None,valid_X=None,valid_y=None):
        if self.num_it_epoch is not None and self.epoch is not None:
            self.num_it_total=self.epoch*self.num_it_epoch
        if self.num_it_total is not None and self.epoch is not None:
            self.num_it_epoch=ceil(self.num_it_total/self.epoch)
        if self.num_it_total is not None and self.num_it_epoch is not None:
            self.epoch=ceil(self.num_it_total/self.num_it_epoch)
        if isinstance(X,SemiTrainDataset):
            self.train_dataset=X

        elif isinstance(X,Dataset) and y is None:
            self.train_dataset.init_dataset(labled_dataset=X, unlabled_dataset=unlabled_X)
        else:
            self.train_dataset.init_dataset(labled_X=X, labled_y=y,unlabled_X=unlabled_X)
        labled_dataloader,unlabled_dataloader=self.train_dataloader.get_dataloader(dataset=self.train_dataset,mu=self.mu)
        # print(self.num_it_total)
        # print(self.num_it_epoch)
        # print(self.epoch)
        self.start_fit()
        self.it_total=0
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
                self.backward(loss)
                self.end_batch_train()
                self.it_total+=1
                self.it_epoch+=1
                print(self.it_total)
                if self.eval_it is not None and _epoch % self.eval_it == 0:
                    self.evaluate(X=valid_X, y=valid_y)
            self.end_epoch()
            if self.eval_epoch is not None and _epoch%self.eval_epoch==0:
                self.evaluate(X=valid_X,y=valid_y)
        self.end_fit()
        return self

    def predict(self,X=None):
        if isinstance(X,Dataset):
            self.test_dataset=X
        else:
            self.test_dataset.init_dataset(X=X)
        test_dataloader=self.test_dataloader.get_dataloader(self.test_dataset)
        self.y_est=torch.Tensor([])
        self.start_predict()
        with torch.no_grad():
            for X,_ in test_dataloader:
                self.start_batch_test()
                self.y_est=torch.cat((self.y_est,self.estimate(X)),0)
                self.end_batch_test()
            self.y_pred=self.get_predict_result(self.y_est)
            self.end_predict()
        return self.y_pred

    def evaluate(self,X,y=None):
        if isinstance(X,Dataset) and y is None:
            y=X.get_y()
        y_pred=self.predict(X)
        y_score=self.y_score
        #print(y_score.shape)
        print(y[:10])
        print(y_pred[:10])
        print(y_score[:10])


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
                print(result[key])
            return result
        else:
            result=self.evaluation.scoring(y,y_pred,y_score)
            return result

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









