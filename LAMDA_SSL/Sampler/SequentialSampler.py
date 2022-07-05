from LAMDA_SSL.Sampler.BaseSampler import BaseSampler
from torch.utils.data import sampler
class SequentialSampler(BaseSampler):
    def __init__(self):
        super().__init__()
    def init_sampler(self,data_source):
        return sampler.SequentialSampler(data_source=data_source)