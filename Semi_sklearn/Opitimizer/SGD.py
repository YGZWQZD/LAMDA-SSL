import torch
from Semi_sklearn.Opitimizer.SemiOptimizer import SemiOptimizer
from torch.optim.optimizer import Optimizer
from torch.optim import  sgd

class SGD(SemiOptimizer):
    def __init__(self, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        self.lr=lr
        self.momentum=momentum
        self.dampening=dampening
        self.weight_decay=weight_decay
        self.nesterov=nesterov
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(defaults)

    def init_optimizer(self,params):
        return sgd.SGD(params=params, lr=self.lr, momentum=self.momentum, dampening=self.dampening,
                   weight_decay=self.weight_decay, nesterov=self.nesterov)






