from torch.optim import lr_scheduler
from LAMDA_SSL.Scheduler.BaseScheduler import BaseScheduler
class CosineAnnealingLR(BaseScheduler):
    def __init__(self,  T_max, eta_min=0, last_epoch=-1, verbose=False):
        super().__init__(last_epoch=last_epoch,verbose=verbose)
        self.T_max=T_max
        self.eta_min=eta_min
    def init_scheduler(self,optimizer):
        return lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.T_max,eta_min=self.eta_min,
                                              last_epoch=self.last_epoch,verbose=self.verbose)


