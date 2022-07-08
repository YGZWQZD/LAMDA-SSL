import  torch.utils.data.distributed as dt
from LAMDA_SSL.Sampler.BaseSampler import BaseSampler
class DistributedSampler(BaseSampler):
    def __init__(self,num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):

        self.num_replicas=num_replicas
        self.rank=rank
        self.shuffle=shuffle
        self.seed=seed
        self.drop_last=drop_last
        super().__init__()

    def init_sampler(self,data_source):
        # >> init_sampler(data_source):  Initialize the sampler with data.
        # >> - data_source: The data to be sampled.
        return dt.DistributedSampler(dataset=data_source,num_replicas=self.num_replicas,rank=self.rank,
                                     shuffle=self.shuffle,seed=self.seed,drop_last=self.drop_last)