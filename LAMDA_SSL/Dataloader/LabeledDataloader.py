from torch.utils.data.dataloader import DataLoader
from LAMDA_SSL.Base.BaseSampler import BaseSampler
from LAMDA_SSL.Sampler.BatchSampler import BatchSampler
class LabeledDataLoader:
    def __init__(self,
                 batch_size= 1, shuffle: bool = False,
                 sampler = None, batch_sampler= None,
                 num_workers: int = 0, collate_fn= None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn = None,
                 multiprocessing_context=None, generator=None,
                 prefetch_factor: int = 2, persistent_workers: bool = False):
        # >> Parameter
        # >> - batch_size: How many samples per batch to load.
        # >> - shuffle: Whether to shuffle the data.
        # >> - sampler: The sampler used when loading data.
        # >> - batch_sampler: set to True to have the data reshuffled at every epoch.
        # >> - num_workers: How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
        # >> - collate_fn: Merges a list of samples to form a mini-batch of Tensor(s).  Used when using batched loading from a map-style dataset.
        # >> - pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.  If your data elements are a custom type, or your :attr:'collate_fn' returns a batch that is a custom type, see the example below.
        # >> - drop_last: Whether to discard redundant data that is not enough for a batch.
        # >> - timeout: If positive, the timeout value for collecting a batch from workers. Should always be non-negative.
        # >> - worker_init_fn: If not None, this will be called on each worker subprocess with the worker id (an int in [0, num_workers - 1]) as input, after seeding and before data loading.
        # >> - multiprocessing_context: The context of multiprocessing.
        # >> - generator: If not None, this RNG will be used by RandomSampler to generate random indexes and multiprocessing to generate base_seed for workers.
        # >> - prefetch_factor: Number of samples loaded in advance by each worker. '2' means there will be a total of 2 * num_workers samples prefetched across all workers.
        # >> - persistent_workers: If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the workers 'Dataset' instances alive.
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
        if isinstance(self.sampler,BaseSampler):
            self.sampler=self.sampler.init_sampler(self.dataset)

        if batch_sampler is not None:
            self.batch_sampler=batch_sampler
        if isinstance(self.batch_sampler,BatchSampler):
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
