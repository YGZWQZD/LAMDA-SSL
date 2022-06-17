from torch.optim import lr_scheduler
class SemiScheduler:
    def __init__(self, last_epoch=-1, verbose=False):
        self.last_epoch=last_epoch
        self.verbose=verbose

    def init_scheduler(self,optimizer):
        return lr_scheduler._LRScheduler(optimizer,last_epoch=self.last_epoch,verbose=self.verbose)
