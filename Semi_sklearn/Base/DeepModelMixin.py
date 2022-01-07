from Semi_sklearn.Dataset.LabledDataset import LabledDataset
from Semi_sklearn.Dataset.SemiTrainDataset import SemiTrainDataset
from Semi_sklearn.Data_loader.SemiTrainDataloader import SemiTrainDataLoader
from Semi_sklearn.Data_loader.SemiTestDataloader import SemiTestDataLoader
from abc import abstractmethod
class SemiDeepModelMixin:
    def __init__(self,train_dataset=None,test_dataset=None,
                 train_dataloader=SemiTrainDataLoader(),
                 test_dataloader=SemiTestDataLoader(),
                 augmentations=None,
                 networks=None,
                 epoch=1
                 ):
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        self.train_dataloader=train_dataloader
        self.test_dataloader=test_dataloader
        self.augmentations=augmentations
        self.networks=networks
        self.epoch=epoch
        self.y_est=None

    def fit(self,labled_X=None,labled_y=None,unlabled_X=None,labled_dataset=None,unlabled_dataset=None,train_dataset=None):
        if train_dataset is not None:
            self.train_dataset=train_dataset
        else:
            if labled_X is not None:
                self.train_dataset.init_dataset(labled_X=labled_X,labled_y=labled_y,unlabled_X=unlabled_X)
            elif labled_dataset is not None:
                self.train_dataset.init_dataset(labled_dataset=labled_dataset,unlabled_dataset=unlabled_dataset)
        self.labled_dataloader,self.unlabled_dataloader=self.train_dataloader.get_dataloader(self.train_dataset)

        for _ in range(self.epoch):
            for (lb_X, lb_y), (ulb_X, _) in zip(self.labled_dataloader,self.unlabled_dataloader):
                train_result=self.train(lb_X,lb_y,ulb_X)
                loss=self.get_loss(train_result)
                self.backward(loss)
            self.scheduler()

    def predict(self,test_X=None,test_dataset=None):
        if test_dataset is not None:
            self.test_dataset=test_dataset
        else:
            self.test_dataset.init_dataset(X=test_X)
        self.test_dataloader=self.test_dataloader.get_dataloader(self.test_dataset)
        self.y_est=[]
        for X,_ in self.test_dataloader:
            self.y_est.append(self.estimate(X))
        y_pred=self.get_predict_result(self.y_est)
        return y_pred

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
    def scheduler(self,*args,**kwargs):
        raise NotImplementedError
    @abstractmethod
    def estimate(self,X,*args,**kwargs):
        raise NotImplementedError
    @abstractmethod
    def get_predict_result(self,y_est,*args,**kwargs):
        raise NotImplementedError







