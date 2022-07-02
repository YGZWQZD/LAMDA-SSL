from lamda_ssl.Sampler.BaseSampler import BaseSampler
import torch.utils.data.sampler as torchsampler
class BatchSampler(BaseSampler):
    def __init__(self, batch_size: int, drop_last: bool):
        super().__init__()
        self.batch_size=batch_size
        self.drop_last=drop_last

    def init_sampler(self,sampler):
        return torchsampler.BatchSampler(sampler=sampler,batch_size=self.batch_size,drop_last=self.drop_last)