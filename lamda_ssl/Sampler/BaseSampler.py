from torch.utils.data.sampler import Sampler
class BaseSampler:
    def __init__(self):
        pass
    def init_sampler(self,data_source):
        return Sampler(data_source=data_source)
