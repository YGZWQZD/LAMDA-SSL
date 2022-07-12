from LAMDA_SSL.Base.BaseSampler import BaseSampler
import torch.utils.data.sampler as torchsampler
class BatchSampler(BaseSampler):
    def __init__(self, batch_size: int, drop_last: bool):
        super().__init__()
        # >> Parameter:
        # >> - batch_size: The number of samples in each batch.
        # >> - drop_last: Whether to discard samples less than one batch.
        self.batch_size=batch_size
        self.drop_last=drop_last

    def init_sampler(self,sampler):
        # >> init_sampler(sampler): Initialize batch sampler with sampler.
        # >> sampler: The sampler used to initial batch sampler.
        return torchsampler.BatchSampler(sampler=sampler,batch_size=self.batch_size,drop_last=self.drop_last)