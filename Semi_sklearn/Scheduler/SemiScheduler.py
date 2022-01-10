from torch.optim.lr_scheduler import LambdaLR
class SemiLambdaLR:
    def __init__(self,  lr_lambda, last_epoch=-1):
        self.lr_lambda=lr_lambda
        self.last_epoch=last_epoch

    def init_scheduler(self,optimizer):
        return LambdaLR(optimizer=optimizer,lr_lambda=self.lr_lambda,last_epoch=self.last_epoch)
