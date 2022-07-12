from LAMDA_SSL.Base.LambdaLR import LambdaLR

class LinearWarmup(LambdaLR):
    def __init__(self,
                 num_training_steps,
                 num_warmup_steps=0,
                 start_factor=0,
                 end_factor=1,
                 last_epoch=-1,
                 verbose=False):
        # >> Parameter:
        # >> - num_training_steps: The total number of iterations for training.
        # >> - num_warmup_steps: The number of iterations to warm up.
        # >> - start_factor: The initialchange factor of the learning rate.
        # >> - end_factor: The final change factor of the learning rate.
        # >> - last_epoch: The index of the last epoch.
        # >> - verbose: Whether to output redundant information.
        self.start_factor=start_factor
        self.end_factor=end_factor
        self.num_warmup_steps=num_warmup_steps
        self.num_training_steps=num_training_steps
        self.verbose=verbose
        super().__init__(lr_lambda=self._lr_lambda,last_epoch=last_epoch,verbose=self.verbose)

    def _lr_lambda(self,current_step):
        if current_step > self.num_warmup_steps:
            return  self.start_factor+float(self.num_training_steps - current_step) \
                    / (self.num_training_steps - self.num_warmup_steps)*(self.end_factor-self.start_factor)
        return 1


