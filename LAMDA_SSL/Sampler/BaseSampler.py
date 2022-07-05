from torch.utils.data.sampler import Sampler
class BaseSampler:
    def __init__(self):
        pass
    def init_sampler(self,data_source):
        # >> init_sampler(data_source):  Initialize the sampler with data.
        # >> - data_source: The data to be sampled.
        return Sampler(data_source=data_source)
