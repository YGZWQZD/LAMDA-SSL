from LAMDA_SSL.Base.BaseSampler import BaseSampler
from torch.utils.data import sampler
class RandomSampler(BaseSampler):
    def __init__(self,replacement: bool = False,
                 num_samples = None, generator=None):
        # >> Parameter:
        # >> - replacement: samples are drawn on-demand with replacement if True.
        # >> - num_samples: The number of samples
        # >> - generator: Generator used in sampling.

        super().__init__()
        self.replacement=replacement
        self.num_samples=num_samples
        self.generator=generator
    def init_sampler(self,data_source):
        # >> init_sampler(data_source):  Initialize the sampler with data.
        # >> - data_source: The data to be sampled.
        return sampler.RandomSampler(data_source=data_source,replacement=self.replacement,
                                     num_samples=self.num_samples,generator=self.generator)