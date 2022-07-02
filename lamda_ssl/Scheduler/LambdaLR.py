from torch.optim import lr_scheduler
from lamda_ssl.Scheduler.BaseScheduler import BaseScheduler
class LambdaLR(BaseScheduler):
    def __init__(self,  lr_lambda, last_epoch=-1,verbose=False):
        super().__init__(last_epoch=last_epoch,verbose=verbose)
        self.lr_lambda = lr_lambda
        self.verbose=verbose

    def init_scheduler(self,optimizer):
        return lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=self.lr_lambda,last_epoch=self.last_epoch,verbose=self.verbose)
