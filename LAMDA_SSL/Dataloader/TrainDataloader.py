import copy

from torch.utils.data.dataloader import DataLoader
from LAMDA_SSL.Base.BaseSampler import BaseSampler
from LAMDA_SSL.Sampler.BatchSampler import BatchSampler

class TrainDataLoader:
    def __init__(self,
                 batch_size=1,
                 shuffle = False, sampler = None,
                 batch_sampler=None,
                 num_workers = 0, collate_fn = None,
                 pin_memory = False, drop_last = True,
                 timeout = 0, worker_init_fn = None,
                 multiprocessing_context=None, generator=None,
                 prefetch_factor = 2,
                 persistent_workers= False,
                 batch_size_adjust=False,labeled_dataloader=None,unlabeled_dataloader=None):
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
        # >> - batch_size_adjust: Whether to automatically adjust the batch_size of labeled_dataloader and unlabeled_dataloader according to the ratio of unlabeled samples to labeled samples.
        # >> - labeled_dataloader: The dataloader of labeled data.
        # >> - unlabeled_dataloader: The dataloader of unlabeled data.
        self.labeled_dataloader=labeled_dataloader
        self.unlabeled_dataloader=unlabeled_dataloader
        if self.labeled_dataloader is None and self.unlabeled_dataloader is None:
            self.batch_size=batch_size
            if isinstance(self.batch_size,(list,tuple)):
                self.labeled_batch_size,self.unlabeled_batch_size=self.batch_size[0],self.batch_size[1]
            elif isinstance(self.batch_size,dict):
                self.labeled_batch_size, self.unlabeled_batch_size = self.batch_size['labeled'], self.batch_size['unlabeled']
            else:
                self.labeled_batch_size, self.unlabeled_batch_size=copy.copy(self.batch_size), copy.copy(self.batch_size)

            self.shuffle=shuffle
            if isinstance(self.shuffle,(list,tuple)):
                self.labeled_shuffle,self.unlabeled_shuffle=self.shuffle[0],self.shuffle[1]
            elif isinstance(self.shuffle,dict):
                self.labeled_shuffle,self.unlabeled_shuffle = self.shuffle['labeled'], self.shuffle['unlabeled']
            else:
                self.labeled_shuffle,self.unlabeled_shuffle=copy.copy(self.shuffle),copy.copy( self.shuffle)

            self.sampler=sampler
            if isinstance(self.sampler,(list,tuple)):
                self.labeled_sampler,self.unlabeled_sampler=self.sampler[0],self.sampler[1]
            elif isinstance(self.sampler,dict):
                self.labeled_sampler,self.unlabeled_sampler = self.sampler['labeled'], self.sampler['unlabeled']
            else:
                self.labeled_sampler,self.unlabeled_sampler=copy.copy(self.sampler), copy.copy(self.sampler)

            self.batch_sampler=batch_sampler
            if isinstance(self.batch_sampler,(list,tuple)):
                self.labeled_batch_sampler,self.unlabeled_batch_sampler=self.batch_sampler[0],self.batch_sampler[1]
            elif isinstance(self.batch_sampler,dict):
                self.labeled_batch_sampler,self.unlabeled_batch_sampler = self.batch_sampler['labeled'], self.batch_sampler['unlabeled']
            else:
                self.labeled_batch_sampler,self.unlabeled_batch_sampler=copy.copy(self.batch_sampler), copy.copy(self.batch_sampler)

            self.num_workers=num_workers
            if isinstance(self.num_workers,(list,tuple)):
                self.labeled_num_workers,self.unlabeled_num_workers=self.num_workers[0],self.num_workers[1]
            elif isinstance(self.num_workers,dict):
                self.labeled_num_workers,self.unlabeled_num_workers = self.num_workers['labeled'], self.num_workers['unlabeled']
            else:
                self.labeled_num_workers,self.unlabeled_num_workers=copy.copy(self.num_workers), copy.copy(self.num_workers)

            self.collate_fn=collate_fn
            if isinstance(self.collate_fn,(list,tuple)):
                self.labeled_collate_fn,self.unlabeled_collate_fn=self.collate_fn[0],self.collate_fn[1]
            elif isinstance(self.collate_fn,dict):
                self.labeled_collate_fn,self.unlabeled_collate_fn= self.collate_fn['labeled'], self.collate_fn['unlabeled']
            else:
                self.labeled_collate_fn,self.unlabeled_collate_fn=copy.copy(self.collate_fn), copy.copy(self.collate_fn)

            self.pin_memory=pin_memory
            if isinstance(self.pin_memory,(list,tuple)):
                self.labeled_pin_memory,self.unlabeled_pin_memory=self.pin_memory[0],self.pin_memory[1]
            elif isinstance(self.pin_memory,dict):
                self.labeled_pin_memory,self.unlabeled_pin_memory = self.pin_memory['labeled'], self.pin_memory['unlabeled']
            else:
                self.labeled_pin_memory,self.unlabeled_pin_memory=copy.copy(self.pin_memory), copy.copy(self.pin_memory)

            self.drop_last=drop_last
            if isinstance(self.drop_last,(list,tuple)):
                self.labeled_drop_last,self.unlabeled_drop_last=self.drop_last[0],self.drop_last[1]
            elif isinstance(self.drop_last,dict):
                self.labeled_drop_last,self.unlabeled_drop_last = self.drop_last['labeled'], self.drop_last['unlabeled']
            else:
                self.labeled_drop_last,self.unlabeled_drop_last=copy.copy(self.drop_last), copy.copy(self.drop_last)

            self.timeout=timeout
            if isinstance(self.timeout,(list,tuple)):
                self.labeled_timeout,self.unlabeled_timeout=self.timeout[0],self.timeout[1]
            elif isinstance(self.timeout,dict):
                self.labeled_timeout,self.unlabeled_timeout = self.timeout['labeled'], self.timeout['unlabeled']
            else:
                self.labeled_timeout,self.unlabeled_timeout=copy.copy(self.timeout), copy.copy(self.timeout)

            self.worker_init_fn=worker_init_fn
            if isinstance(self.worker_init_fn,(list,tuple)):
                self.labeled_worker_init_fn,self.unlabeled_worker_init_fn=self.worker_init_fn[0],self.worker_init_fn[1]
            elif isinstance(self.worker_init_fn,dict):
                self.labeled_worker_init_fn,self.unlabeled_worker_init_fn = self.worker_init_fn['labeled'], self.worker_init_fn['unlabeled']
            else:
                self.labeled_worker_init_fn,self.unlabeled_worker_init_fn=copy.copy(self.worker_init_fn), copy.copy(self.worker_init_fn)

            self.multiprocessing_context=multiprocessing_context
            if isinstance(self.multiprocessing_context,(list,tuple)):
                self.labeled_multiprocessing_context,self.unlabeled_multiprocessing_context=self.multiprocessing_context[0],self.multiprocessing_context[1]
            elif isinstance(self.multiprocessing_context,dict):
                self.labeled_multiprocessing_context,self.unlabeled_multiprocessing_context = self.multiprocessing_context['labeled'], self.multiprocessing_context['unlabeled']
            else:
                self.labeled_multiprocessing_context,self.unlabeled_multiprocessing_context=copy.copy(self.multiprocessing_context), copy.copy(self.multiprocessing_context)

            self.generator=generator
            if isinstance(self.generator,(list,tuple)):
                self.labeled_generator,self.unlabeled_generator=self.generator[0],self.generator[1]
            elif isinstance(self.generator,dict):
                self.labeled_generator,self.unlabeled_generator = self.generator['labeled'], self.generator['unlabeled']
            else:
                self.labeled_generator,self.unlabeled_generator=copy.copy(self.generator), copy.copy(self.generator)

            self.prefetch_factor=prefetch_factor
            if isinstance(self.prefetch_factor,(list,tuple)):
                self.labeled_prefetch_factor,self.unlabeled_prefetch_factor=self.prefetch_factor[0],self.prefetch_factor[1]
            elif isinstance(self.prefetch_factor,dict):
                self.labeled_prefetch_factor,self.unlabeled_prefetch_factor = self.prefetch_factor['labeled'], self.prefetch_factor['unlabeled']
            else:
                self.labeled_prefetch_factor,self.unlabeled_prefetch_factor=copy.copy(self.prefetch_factor), copy.copy(self.prefetch_factor)

            self.persistent_workers=persistent_workers
            if isinstance(self.persistent_workers,(list,tuple)):
                self.labeled_persistent_workers,self.unlabeled_persistent_workers=self.persistent_workers[0],self.persistent_workers[1]
            elif isinstance(self.persistent_workers,dict):
                self.labeled_persistent_workers,self.unlabeled_persistent_workers = self.persistent_workers['labeled'], self.persistent_workers['unlabeled']
            else:
                self.labeled_persistent_workers,self.unlabeled_persistent_workers=copy.copy(self.persistent_workers), copy.copy(self.persistent_workers)
        else:
            self.labeled_batch_size, self.unlabeled_batch_size = self.labeled_dataloader.batch_size, self.unlabeled_dataloader.batch_size
            self.batch_size=[self.labeled_batch_size, self.unlabeled_batch_size]

            self.labeled_shuffle, self.unlabeled_shuffle = self.labeled_dataloader.shuffle, self.unlabeled_dataloader.shuffle
            self.shuffle = [self.labeled_shuffle, self.unlabeled_shuffle ]

            self.labeled_sampler, self.unlabeled_sampler = self.labeled_dataloader.sampler, self.unlabeled_dataloader.sampler
            self.sampler = [self.labeled_sampler, self.unlabeled_sampler]

            self.labeled_batch_sampler, self.unlabeled_batch_sampler = self.labeled_dataloader.batch_sampler, self.unlabeled_dataloader.batch_sampler
            self.batch_sampler = [self.labeled_batch_sampler, self.unlabeled_batch_sampler]


            self.labeled_num_workers, self.unlabeled_num_workers = self.labeled_dataloader.num_workers, self.unlabeled_dataloader.num_workers
            self.num_workers = [self.labeled_num_workers, self.unlabeled_num_workers ]

            self.labeled_collate_fn, self.unlabeled_collate_fn = self.labeled_dataloader.collate_fn, self.unlabeled_dataloader.collate_fn
            self.collate_fn = [self.labeled_collate_fn, self.unlabeled_collate_fn ]

            self.labeled_pin_memory, self.unlabeled_pin_memory = self.labeled_dataloader.pin_memory, self.unlabeled_dataloader.pin_memory
            self.pin_memory = [self.labeled_pin_memory, self.unlabeled_pin_memory ]

            self.labeled_drop_last, self.unlabeled_drop_last = self.labeled_dataloader.drop_last, self.unlabeled_dataloader.drop_last
            self.drop_last= [self.labeled_drop_last, self.unlabeled_drop_last ]

            self.labeled_timeout, self.unlabeled_timeout = self.labeled_dataloader.timeout, self.unlabeled_dataloader.timeout
            self.timeout= [self.labeled_timeout, self.unlabeled_timeout]

            self.labeled_worker_init_fn , self.unlabeled_worker_init_fn  = self.labeled_dataloader.worker_init_fn , self.unlabeled_dataloader.worker_init_fn
            self.worker_init_fn = [self.labeled_worker_init_fn , self.unlabeled_worker_init_fn ]

            self.labeled_multiprocessing_context , self.unlabeled_multiprocessing_context  = self.labeled_dataloader.multiprocessing_context , self.unlabeled_dataloader.multiprocessing_context
            self.multiprocessing_context = [self.labeled_multiprocessing_context , self.unlabeled_multiprocessing_context ]

            self.labeled_generator , self.unlabeled_generator  = self.labeled_dataloader.generator , self.unlabeled_dataloader.generator
            self.generator = [self.labeled_generator , self.unlabeled_generator ]

            self.labeled_prefetch_factor , self.unlabeled_prefetch_factor  = self.labeled_dataloader.prefetch_factor , self.unlabeled_dataloader.prefetch_factor
            self.prefetch_factor = [self.labeled_prefetch_factor , self.unlabeled_prefetch_factor ]

            self.labeled_persistent_workers , self.unlabeled_persistent_workers  = self.labeled_dataloader.persistent_workers , self.unlabeled_dataloader.persistent_workers
            self.persistent_workers = [self.labeled_persistent_workers , self.unlabeled_persistent_workers ]

        self.dataset=None
        self.labeled_dataset=None
        self.unlabeled_dataset=None
        self.len_labeled=None
        self.len_unlabeled=None
        self.batch_size_adjust=batch_size_adjust

    def init_dataloader(self,dataset=None,labeled_dataset=None,unlabeled_dataset=None,sampler=None,batch_sampler=None,mu=None):

        if mu is not None and self.labeled_batch_size is not None:
            self.mu=mu
            self.unlabeled_batch_size=self.mu*self.labeled_batch_size

        if dataset is not None:
            self.labeled_dataset=dataset.labeled_dataset
            self.unlabeled_dataset=dataset.unlabeled_dataset
        elif labeled_dataset is not None and unlabeled_dataset is not None:
            self.labeled_dataset=labeled_dataset
            self.unlabeled_dataset=unlabeled_dataset
        else:
            raise ValueError('No dataset')

        self.len_labeled=self.labeled_dataset.__len__()
        self.len_unlabeled=self.unlabeled_dataset.__len__()

        if self.batch_size_adjust:
            if self.len_labeled < self.len_unlabeled:
                self.unlabeled_batch_size=self.labeled_batch_size*(self.len_unlabeled//self.len_labeled)
            else:
                self.labeled_batch_size = self.unlabeled_batch_size * (self.len_labeled//self.len_unlabeled)
            self.mu=self.len_labeled//self.len_unlabeled

        if sampler is not None:
            if isinstance(sampler,(list,tuple)):
                self.labeled_sampler,self.unlabeled_sampler=sampler[0],sampler[1]
            elif isinstance(sampler,dict):
                self.labeled_sampler,self.unlabeled_sampler=sampler['labeled'],sampler['unlabeled']
            else:
                self.labeled_sampler, self.unlabeled_sampler=copy.copy(sampler),copy.copy(sampler)
            if self.mu is not None:
                if self.labeled_sampler is not None and self.unlabeled_sampler is not None and\
                        hasattr(self.labeled_sampler, 'num_samples') and hasattr(self.unlabeled_sampler, 'num_samples')  \
                        and self.labeled_sampler.num_samples is not None and self.unlabeled_sampler.replacement is True:
                    self.unlabeled_sampler.num_samples = self.labeled_sampler.num_samples *self.mu

        if batch_sampler is not None:
            if isinstance(batch_sampler,(list,tuple)):
                self.labeled_batch_sampler,self.unlabeled_batch_sampler=batch_sampler[0],batch_sampler[1]
            elif isinstance(batch_sampler,dict):
                self.labeled_batch_sampler,self.unlabeled_batch_sampler=batch_sampler['labeled'],batch_sampler['unlabeled']
            else:
                self.labeled_batch_sampler, self.unlabeled_batch_sampler=copy.copy(batch_sampler),copy.copy(batch_sampler)
            if self.mu is not None:
                if self.labeled_batch_sampler is not None and self.unlabeled_batch_sampler is not None and\
                        hasattr(self.labeled_batch_sampler, 'batch_size') and hasattr(self.unlabeled_batch_sampler, 'batch_size')\
                        and self.labeled_batch_sampler.batch_size is not None:
                    self.unlabeled_batch_sampler.batch_size = self.labeled_batch_sampler.batch_size*self.mu

        if isinstance(self.labeled_sampler,BaseSampler):
            self.labeled_sampler=self.labeled_sampler.init_sampler(self.labeled_dataset)

        if isinstance(self.labeled_batch_sampler,BatchSampler):
            self.labeled_batch_sampler=self.labeled_batch_sampler.init_sampler(self.labeled_sampler)

        if isinstance(self.unlabeled_sampler,BaseSampler):
            self.unlabeled_sampler=self.unlabeled_sampler.init_sampler(self.unlabeled_dataset)

        if isinstance(self.unlabeled_batch_sampler,BatchSampler):
            self.unlabeled_batch_sampler=self.unlabeled_batch_sampler.init_sampler(self.unlabeled_sampler)

        # if self.labeled_dataloader is not None and self.unlabeled_dataloader is not None:
        #     self.labeled_dataloader.batch_size=self.labeled_batch_size
        #     self.unlabeled_dataloader.batch_size=self.unlabeled_batch_size
        #
        #     if self.mu is not None:
        #         if self.labeled_dataloader.sampler is not None and self.unlabeled_dataloader.sampler is not None\
        #                 and hasattr(self.labeled_dataloader.sampler,'num_samples')\
        #                 and hasattr(self.unlabeled_dataloader.sampler,'num_samples')\
        #                 and self.labeled_dataloader.sampler.num_samples is not None:
        #             self.unlabeled_dataloader.sampler.num_samples = self.labeled_dataloader.sampler.num_samples*self.mu
        #     if self.mu is not None:
        #         if self.labeled_dataloader.batch_sampler is not None  \
        #                 and self.unlabeled_dataloader.batch_sampler is not None\
        #                 and hasattr(self.labeled_dataloader.batch_sampler,'batch_size')\
        #                 and hasattr(self.unlabeled_dataloader.batch_sampler,'batch_size')\
        #                 and self.labeled_dataloader.batch_sampler.batch_size is not None:
        #             self.unlabeled_dataloader.batch_sampler.batch_size = self.labeled_dataloader.batch_sampler.batch_size*self.mu
        #
        #
        #     return self.labeled_dataloader,self.unlabeled_dataloader


        if self.labeled_batch_sampler is None and self.labeled_sampler is None:
            if self.labeled_dataloader is None:
                self.labeled_dataloader=DataLoader(dataset=self.labeled_dataset,
                                    batch_size=self.labeled_batch_size,
                                    shuffle = self.labeled_shuffle,
                                    sampler = self.labeled_sampler,
                                    batch_sampler = self.labeled_batch_sampler,
                                    num_workers = self.labeled_num_workers,
                                    collate_fn = self.labeled_collate_fn,
                                    pin_memory = self.labeled_pin_memory,
                                    drop_last = self.labeled_drop_last,
                                    timeout = self.labeled_timeout,
                                    worker_init_fn = self.labeled_worker_init_fn,
                                    multiprocessing_context = self.labeled_multiprocessing_context,
                                    generator = self.labeled_generator,
                                    prefetch_factor = self.labeled_prefetch_factor,
                                    persistent_workers = self.labeled_persistent_workers)
            else:
                self.labeled_dataloader.batch_size = self.labeled_batch_size
                self.labeled_dataloader=self.labeled_dataloader.init_dataloader(dataset=self.labeled_dataset)


        elif self.labeled_batch_sampler is not None:
            if self.labeled_dataloader is None:
                self.labeled_dataloader=DataLoader(dataset=self.labeled_dataset,
                                    batch_sampler = self.labeled_batch_sampler,
                                    num_workers = self.labeled_num_workers,
                                    collate_fn = self.labeled_collate_fn,
                                    pin_memory = self.labeled_pin_memory,
                                    timeout = self.labeled_timeout,
                                    worker_init_fn = self.labeled_worker_init_fn,
                                    multiprocessing_context = self.labeled_multiprocessing_context,
                                    generator = self.labeled_generator,
                                    prefetch_factor = self.labeled_prefetch_factor,
                                    persistent_workers = self.labeled_persistent_workers)
            else:
                self.labeled_dataloader=self.labeled_dataloader.init_dataloader(dataset=self.labeled_dataset,
                                                                                batch_sampler=self.labeled_batch_sampler)

        else:
            if self.labeled_dataloader is None:
                self.labeled_dataloader=DataLoader(dataset=self.labeled_dataset,
                                    batch_size=self.labeled_batch_size,
                                    shuffle = False,
                                    sampler = self.labeled_sampler,
                                    num_workers = self.labeled_num_workers,
                                    collate_fn = self.labeled_collate_fn,
                                    pin_memory = self.labeled_pin_memory,
                                    drop_last = self.labeled_drop_last,
                                    timeout = self.labeled_timeout,
                                    worker_init_fn = self.labeled_worker_init_fn,
                                    multiprocessing_context = self.labeled_multiprocessing_context,
                                    generator = self.labeled_generator,
                                    prefetch_factor = self.labeled_prefetch_factor,
                                    persistent_workers = self.labeled_persistent_workers)
            else:
                self.labeled_dataloader.batch_size = self.labeled_batch_size
                self.labeled_dataloader = self.labeled_dataloader.init_dataloader(dataset=self.labeled_dataset,
                                                                                  sampler=self.labeled_sampler)


        if self.unlabeled_batch_sampler is None and self.unlabeled_sampler is None:
            if self.unlabeled_dataloader is None:
                self.unlabeled_dataloader=DataLoader(dataset=self.unlabeled_dataset,
                                    batch_size=self.unlabeled_batch_size,
                                    shuffle = self.unlabeled_shuffle,
                                    sampler = self.unlabeled_sampler,
                                    batch_sampler = self.unlabeled_batch_sampler,
                                    num_workers = self.unlabeled_num_workers,
                                    collate_fn = self.unlabeled_collate_fn,
                                    pin_memory = self.unlabeled_pin_memory,
                                    drop_last = self.unlabeled_drop_last,
                                    timeout = self.unlabeled_timeout,
                                    worker_init_fn = self.unlabeled_worker_init_fn,
                                    multiprocessing_context = self.unlabeled_multiprocessing_context,
                                    generator = self.unlabeled_generator,
                                    prefetch_factor = self.unlabeled_prefetch_factor,
                                    persistent_workers = self.unlabeled_persistent_workers)
            else:
                self.unlabeled_dataloader.batch_size = self.unlabeled_batch_size
                self.unlabeled_dataloader=self.unlabeled_dataloader.init_dataloader(dataset=self.unlabeled_dataset)


        elif self.unlabeled_batch_sampler is not None:
            if self.unlabeled_dataloader is None:
                self.unlabeled_dataloader=DataLoader(dataset=self.unlabeled_dataset,
                                    batch_sampler = self.unlabeled_batch_sampler,
                                    num_workers = self.unlabeled_num_workers,
                                    collate_fn = self.unlabeled_collate_fn,
                                    pin_memory = self.unlabeled_pin_memory,
                                    timeout = self.unlabeled_timeout,
                                    worker_init_fn = self.unlabeled_worker_init_fn,
                                    multiprocessing_context = self.unlabeled_multiprocessing_context,
                                    generator = self.unlabeled_generator,
                                    prefetch_factor = self.unlabeled_prefetch_factor,
                                    persistent_workers = self.unlabeled_persistent_workers)
            else:
                self.unlabeled_dataloader = self.unlabeled_dataloader.init_dataloader(dataset=self.unlabeled_dataset,
                                                                                  batch_sampler=self.unlabeled_batch_sampler)
        else:
            if self.unlabeled_dataloader is None:
                self.unlabeled_dataloader=DataLoader(dataset=self.unlabeled_dataset,
                                    batch_size=self.unlabeled_batch_size,
                                    shuffle = False,
                                    sampler = self.unlabeled_sampler,
                                    num_workers = self.unlabeled_num_workers,
                                    collate_fn = self.unlabeled_collate_fn,
                                    pin_memory = self.unlabeled_pin_memory,
                                    drop_last = self.unlabeled_drop_last,
                                    timeout = self.unlabeled_timeout,
                                    worker_init_fn = self.unlabeled_worker_init_fn,
                                    multiprocessing_context = self.unlabeled_multiprocessing_context,
                                    generator = self.unlabeled_generator,
                                    prefetch_factor = self.unlabeled_prefetch_factor,
                                    persistent_workers = self.unlabeled_persistent_workers)
            else:
                self.unlabeled_dataloader.batch_size = self.unlabeled_batch_size
                self.unlabeled_dataloader = self.unlabeled_dataloader.init_dataloader(dataset=self.unlabeled_dataset,
                                                                                  sampler=self.unlabeled_sampler)


        return self.labeled_dataloader,self.unlabeled_dataloader

