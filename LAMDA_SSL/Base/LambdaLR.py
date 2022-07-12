from torch.optim import lr_scheduler
from LAMDA_SSL.Base.BaseScheduler import BaseScheduler
class LambdaLR(BaseScheduler):
    def __init__(self,  lr_lambda, last_epoch=-1,verbose=False):
        # >> Parameter:
        # >> - lr_lambda: A function which computes a multiplicative factor given an integer parameter epoch, or a list of such functions, one for each group in optimizer.param_groups.
        # >> - last_epoch: The index of last epoch.
        # >> - verbose: If 'True', prints a message to stdout for each update.
        super().__init__(last_epoch=last_epoch,verbose=verbose)
        self.lr_lambda = lr_lambda
        self.verbose=verbose

    def init_scheduler(self,optimizer):
        return lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=self.lr_lambda,last_epoch=self.last_epoch,verbose=self.verbose)
