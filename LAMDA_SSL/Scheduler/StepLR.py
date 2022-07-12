from torch.optim import lr_scheduler
from LAMDA_SSL.Base.BaseScheduler import BaseScheduler
class StepLR(BaseScheduler):
    def __init__(self,  step_size, gamma=0.1, last_epoch=-1, verbose=False):
        # >> Parameter:
        # >> - step_size: Period of learning rate decay.
        # >> - gamma: Multiplicative factor of learning rate decay.
        # >> - last_epoch: The index of last epoch.
        # >> - verbose: If 'True', prints a message to stdout for each update.
        super().__init__(last_epoch=last_epoch,verbose=verbose)
        self.step_size=step_size
        self.gamma=gamma
        self.last_epoch=last_epoch
        self.verbose=verbose

    def init_scheduler(self,optimizer):
        return lr_scheduler.StepLR(optimizer=optimizer,step_size=self.step_size,gamma=self.gamma,last_epoch=self.last_epoch,verbose=self.verbose)
