from LAMDA_SSL.Base.BaseOptimizer import BaseOptimizer
from torch.optim import adam
class Adam(BaseOptimizer):
    def __init__(self,lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        # >> Parameter:
        # >> - lr: learning rate.
        # >> - betas: Coefficients used for computing running averages of gradient and its square.
        # >> - eps: Term added to the denominator to improve numerical stability
        # >> - weight_decay: Weight decay (L2 penalty)
        # >> - amsgrad: whether to use the AMSGrad variant of this algorithm from the paper 'On the Convergence of Adam and Beyond'.
        self.lr=lr
        self.betas=betas
        self.eps=eps
        self.weight_decay=weight_decay
        self.amsgrad=amsgrad
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(defaults=defaults)

    def init_optimizer(self,params):
        return adam.Adam(params=params,lr=self.lr,betas=self.betas,eps=self.eps,
                    weight_decay=self.weight_decay,amsgrad=self.amsgrad)


