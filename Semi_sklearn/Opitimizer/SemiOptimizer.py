from torch.optim.optimizer import Optimizer
class SemiOptimizer:
    def __init__(self,defaults):
        self.defaults=defaults
    def init_optimizer(self,params):
        return Optimizer(params=params,default=self.defaults)

