from LAMDA_SSL.Base.BaseSampler import BaseSampler
from torch.utils.data import sampler
class SequentialSampler(BaseSampler):
    def __init__(self):
        super().__init__()
    def init_sampler(self,data_source):
        # >> init_sampler(data_source):  Initialize the sampler with data.
        # >> - data_source: The data to be sampled.
        return sampler.SequentialSampler(data_source=data_source)