from torch.optim import lr_scheduler
from lamda_ssl.Scheduler.BaseScheduler import BaseScheduler
class StepLR(BaseScheduler):
    def __init__(self,  step_size, gamma=0.1, last_epoch=-1, verbose=False):
        super().__init__(last_epoch=last_epoch,verbose=verbose)
        self.step_size=step_size
        self.gamma=gamma
        self.last_epoch=last_epoch
        self.verbose=verbose

    def init_scheduler(self,optimizer):
        return lr_scheduler.StepLR(optimizer=optimizer,step_size=self.step_size,gamma=self.gamma,last_epoch=self.last_epoch,verbose=self.verbose)
