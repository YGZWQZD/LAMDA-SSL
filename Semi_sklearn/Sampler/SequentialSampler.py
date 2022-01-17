from Semi_sklearn.Sampler.SemiSampler import SemiSampler
from torch.utils.data import sampler
class SequentialSampler(SemiSampler):
    def __init__(self):
        super().__init__()
    def init_sampler(self,data_source):
        return sampler.SequentialSampler(data_source=data_source)