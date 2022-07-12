from torch.optim.optimizer import Optimizer
class BaseOptimizer:
    def __init__(self,defaults):
        # >> - defaults: A dict containing default values of optimization options (used when a parameter group doesn't specify them).
        self.defaults=defaults
    def init_optimizer(self,params):
        # >> init_optimizer(params): Put the parameters that need to be optimized into the optimizer.
        # >> - params: The parameters to be optimized.
        return Optimizer(params=params,default=self.defaults)

