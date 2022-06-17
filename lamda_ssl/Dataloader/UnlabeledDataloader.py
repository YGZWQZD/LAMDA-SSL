from torch.utils.data.dataloader import DataLoader
from lamda_ssl.Sampler.SemiSampler import SemiSampler
from lamda_ssl.Sampler.BatchSampler import SemiBatchSampler
class UnlabeledDataLoader:
    def __init__(self,batch_size= 1,
                 shuffle: bool = False, sampler = None,
                 batch_sampler= None,
                 num_workers: int = 0, collate_fn= None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn = None,
                 multiprocessing_context=None, generator=None,
                 prefetch_factor: int = 2,
                 persistent_workers: bool = False):
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.sampler=sampler
        self.batch_sampler=batch_sampler
        self.num_workers=num_workers
        self.collate_fn=collate_fn
        self.pin_memory=pin_memory
        self.drop_last=drop_last
        self.timeout=timeout
        self.worker_init_fn=worker_init_fn
        self.multiprocessing_context=multiprocessing_context
        self.generator=generator
        self.prefetch_factor=prefetch_factor
        self.persistent_workers=persistent_workers
        self.dataset=None
        self.dataloader=None

    def init_dataloader(self,dataset=None,sampler=None,batch_sampler=None):
        self.dataset=dataset
        if sampler is not None:
            self.sampler=sampler
        if isinstance(self.sampler,SemiSampler):
            self.sampler=self.sampler.init_sampler(self.dataset)

        if batch_sampler is not None:
            self.batch_sampler=batch_sampler
        if isinstance(self.batch_sampler,SemiBatchSampler):
            self.batch_sampler=self.batch_sampler.init_sampler(self.sampler)

        if self.batch_sampler is None and self.sampler is None:
            self.dataloader= DataLoader(dataset=self.dataset,
                                batch_size=self.batch_size,
                                shuffle = self.shuffle,
                                sampler = self.sampler,
                                batch_sampler = self.batch_sampler,
                                num_workers = self.num_workers,
                                collate_fn = self.collate_fn,
                                pin_memory = self.pin_memory,
                                drop_last = self.drop_last,
                                timeout = self.timeout,
                                worker_init_fn = self.worker_init_fn,
                                multiprocessing_context = self.multiprocessing_context,
                                generator = self.generator,
                                prefetch_factor = self.prefetch_factor,
                                persistent_workers = self.persistent_workers)
        elif self.batch_sampler is not None:
            self.dataloader = DataLoader(dataset=self.dataset,
                                         batch_sampler=self.batch_sampler,
                                         num_workers=self.num_workers,
                                         collate_fn=self.collate_fn,
                                         pin_memory=self.pin_memory,
                                         timeout=self.timeout,
                                         worker_init_fn=self.worker_init_fn,
                                         multiprocessing_context=self.multiprocessing_context,
                                         generator=self.generator,
                                         prefetch_factor=self.prefetch_factor,
                                         persistent_workers=self.persistent_workers)
        else:
            self.dataloader= DataLoader(dataset=self.dataset,
                                batch_size=self.batch_size,
                                shuffle = False,
                                sampler = self.sampler,
                                num_workers = self.num_workers,
                                collate_fn = self.collate_fn,
                                pin_memory = self.pin_memory,
                                drop_last = self.drop_last,
                                timeout = self.timeout,
                                worker_init_fn = self.worker_init_fn,
                                multiprocessing_context = self.multiprocessing_context,
                                generator = self.generator,
                                prefetch_factor = self.prefetch_factor,
                                persistent_workers = self.persistent_workers)
        return self.dataloader
# a=SemiTestDataLoader()
# print(type(a).__name__)
# print(type(SemiTestDataLoader).__name__)
