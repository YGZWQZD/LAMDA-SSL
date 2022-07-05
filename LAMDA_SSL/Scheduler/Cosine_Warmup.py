from LAMDA_SSL.Scheduler.LambdaLR import LambdaLR
import math
class Cosine_Warmup(LambdaLR):
    def __init__(self,
                 num_training_steps,
                 num_warmup_steps=0,
                 num_cycles=7./16,
                 last_epoch=-1,
                 verbose=False):
        self.num_cycles=num_cycles # upper bound
        self.num_warmup_steps=num_warmup_steps
        self.num_training_steps=num_training_steps
        self.verbose=verbose
        super().__init__(lr_lambda=self._lr_lambda,last_epoch=last_epoch,verbose=self.verbose)

    def _lr_lambda(self,current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        no_progress = float(current_step - self.num_warmup_steps) / \
            float(max(1, self.num_training_steps - self.num_warmup_steps))
        return max(0., math.cos(math.pi * self.num_cycles * no_progress))


