from LAMDA_SSL.Base.BaseOptimizer import BaseOptimizer
from torch.optim import  sgd

class SGD(BaseOptimizer):
    def __init__(self, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        # >> Parameter:
        # >> - lr: Learning rate.
        # >> - momentum: Momentum factor.
        # >> - dampening: Dampening for momentum.
        # >> - weight_decay: Weight decay (L2 penalty).
        # >> - nesterov: Enables Nesterov momentum.
        self.lr=lr
        self.momentum=momentum
        self.dampening=dampening
        self.weight_decay=weight_decay
        self.nesterov=nesterov
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        super().__init__(defaults)

    def init_optimizer(self,params):
        opti=sgd.SGD(params=params, lr=self.lr, momentum=self.momentum, dampening=self.dampening,
                   weight_decay=self.weight_decay, nesterov=self.nesterov)
        return opti





