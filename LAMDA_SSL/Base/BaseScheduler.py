from torch.optim import lr_scheduler
class BaseScheduler:
    def __init__(self, last_epoch=-1, verbose=False):
        # >> Parameter:
        # >> - last_epoch: The index of last epoch.
        # >> - verbose: If 'True', prints a message to stdout for each update.
        self.last_epoch=last_epoch
        self.verbose=verbose

    def init_scheduler(self,optimizer):
        # >> init_scheduler(optimizer): Initialize the scheduler with the optimizer.
        # >> - optimizer: The optimizer used by the model.
        return lr_scheduler._LRScheduler(optimizer,last_epoch=self.last_epoch,verbose=self.verbose)
