from lamda_ssl.Sampler.BaseSampler import BaseSampler
from torch.utils.data import sampler
class RandomSampler(BaseSampler):
    def __init__(self,replacement: bool = False,
                 num_samples = None, generator=None):
        super().__init__()
        self.replacement=replacement
        self.num_samples=num_samples
        self.generator=generator
    def init_sampler(self,data_source):
        # if num_samples is not None:
        #     self.num_samples=num_samples
        return sampler.RandomSampler(data_source=data_source,replacement=self.replacement,
                                     num_samples=self.num_samples,generator=self.generator)