from LAMDA_SSL.Base.LambdaLR import LambdaLR
import math
class CosineWarmup(LambdaLR):
    def __init__(self,
                 num_training_steps,
                 num_warmup_steps=0,
                 num_cycles=7./16,
                 last_epoch=-1,
                 verbose=False):
        # >> Parameter:
        # >> - num_training_steps: The total number of iterations for training.
        # >> - num_warmup_steps: The number of iterations to warm up.
        # >> - num_cycles: The upperbound of the multiplicative factor is num_cycles PI.
        # >> - last_epoch: The index of the last epoch.
        # >> - verbose: Whether to output redundant information.
        self.num_warmup_steps=num_warmup_steps
        self.num_cycles=num_cycles
        self.num_training_steps=num_training_steps
        self.verbose=verbose
        super().__init__(lr_lambda=self._lr_lambda,last_epoch=last_epoch,verbose=self.verbose)

    def _lr_lambda(self,current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        no_progress = float(current_step - self.num_warmup_steps) / \
            float(max(1, self.num_training_steps - self.num_warmup_steps))
        return max(0., math.cos(math.pi * self.num_cycles * no_progress))


