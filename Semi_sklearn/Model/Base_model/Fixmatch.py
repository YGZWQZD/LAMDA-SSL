from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from Semi_sklearn.Base.SemiDeepModelMixin import SemiDeepModelMixin
from Semi_sklearn.Data_loader.SemiTrainDataloader import SemiTrainDataLoader
from Semi_sklearn.Data_loader.SemiTestDataloader import SemiTestDataLoader
class Fixmatch(InductiveEstimator,SemiDeepModelMixin):
    def __init__(self,train_dataset=None,test_dataset=None,
                 train_dataloader=None,
                 test_dataloader=None,
                 augmentation=None,
                 weakly_augmentation=None,
                 strong_augmentation=None,
                 normalization=None,
                 network=None,
                 epoch=1,
                 optimizer=None,
                 scheduler=None,
                 lr=None,
                 device=None,
                 threshold=None,
                 lambda_u=None,
                 mu=None,
                 ema=None
                 ):
        SemiDeepModelMixin.__init__(self,train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    train_dataloader=train_dataloader,
                                    test_dataloader=test_dataloader,
                                    augmentation=augmentation,
                                    network=network,
                                    epoch=epoch,
                                    mu=mu,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    lr=lr,
                                    device=device
                                    )
        self.ema=ema
        self.lambda_u=lambda_u
        self.threshold=threshold
        self.normalization=normalization
        if weakly_augmentation is not None:
            self.weakly_augmentation=weakly_augmentation
            self.strong_augmentation=strong_augmentation
        elif isinstance(self.augmentation,dict):
            self.weakly_augmentation=self.augmentation['weakly_augmentation']
            self.strong_augmentation = self.augmentation['strong_augmentation']
        elif isinstance(self.augmentation,list):
            self.weakly_augmentation = self.augmentation[0]
            self.strong_augmentation = self.augmentation[1]
        elif isinstance(self.augmentation,tuple):
            self.weakly_augmentation,self.strong_augmentation=self.augmentation
        else:
            self.weakly_augmentation = self.augmentation[0]
            self.strong_augmentation = self.augmentation[1]
        self.normalization=self.augmentation['normalization'] if self.augmentation['normalization'] is not None else self.normalization
    def train(self,lb_X,lb_y,ulb_X,*args,**kwargs):
        w_lb_X=self.weakly_augmentation.fit_transform(lb_X)




    def get_loss(self,train_result,*args,**kwargs):
        raise NotImplementedError

    def backward(self,loss,*args,**kwargs):
        raise NotImplementedError

    def scheduler(self,*args,**kwargs):
        raise NotImplementedError

    def estimate(self,X,*args,**kwargs):
        X=self.normalization.fit_transform(X)
        


    def get_predict_result(self,y_est,*args,**kwargs):
        raise NotImplementedError



