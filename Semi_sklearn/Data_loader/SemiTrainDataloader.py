import copy

from torch.utils.data.dataloader import DataLoader
from Semi_sklearn.Sampler.SemiSampler import SemiSampler
from Semi_sklearn.Sampler.BatchSampler import SemiBatchSampler

class SemiTrainDataLoader:
    def __init__(self,
                 batch_size=1,
                 shuffle = False, sampler = None,
                 batch_sampler=None, Iterable = None,
                 num_workers = 0, collate_fn = None,
                 pin_memory = False, drop_last = True,
                 timeout = 0, worker_init_fn = None,
                 multiprocessing_context=None, generator=None,
                 prefetch_factor = 2,
                 persistent_workers= False,
                 batch_size_adjust=False):

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

        self.Iterable=Iterable
        if isinstance(self.Iterable,(list,tuple)):
            self.labeled_Iterable,self.unlabeled_Iterable=self.Iterable[0],self.Iterable[1]
        elif isinstance(self.Iterable,dict):
            self.labeled_Iterable,self.unlabeled_Iterable = self.Iterable['labeled'], self.Iterable['unlabeled']
        else:
            self.labeled_Iterable,self.unlabeled_Iterable = copy.copy(self.Iterable), copy.copy(self.Iterable)

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

        self.dataset=None
        self.labeled_dataset=None
        self.unlabeled_dataset=None
        self.labeled_dataloader=None
        self.unlabeled_dataloader=None
        self.len_labeled=None
        self.len_unlabeled=None
        self.batch_size_adjust=batch_size_adjust

    def init_dataloader(self,dataset=None,labeled_dataset=None,unlabeled_dataset=None,sampler=None,batch_sampler=None,mu=None):
        if dataset is not None:
            self.labeled_dataset=dataset.get_dataset(labeled=True)
            self.unlabeled_dataset=dataset.get_dataset(labeled=False)
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
        if mu is not None and self.labeled_batch_size is not None:
            self.unlabeled_batch_size=mu*self.labeled_batch_size

        if sampler is not None:
            if isinstance(sampler,(list,tuple)):
                self.labeled_sampler,self.unlabeled_sampler=sampler[0],sampler[1]
            elif isinstance(sampler,dict):
                self.labeled_sampler,self.unlabeled_sampler=sampler['labeled'],sampler['unlabeled']
            else:
                self.labeled_sampler, self.unlabeled_sampler=copy.copy(sampler),copy.copy(sampler)
            if mu is not None:
                if hasattr(self.labeled_sampler, 'num_samples') and hasattr(self.unlabeled_sampler, 'num_samples')  \
                        and self.labeled_sampler.num_samples is not None:
                    self.unlabeled_sampler.num_samples = self.labeled_sampler.num_samples * mu

        if batch_sampler is not None:
            if isinstance(batch_sampler,(list,tuple)):
                self.labeled_batch_sampler,self.unlabeled_batch_sampler=batch_sampler[0],batch_sampler[1]
            elif isinstance(batch_sampler,dict):
                self.labeled_batch_sampler,self.unlabeled_batch_sampler=batch_sampler['labeled'],batch_sampler['unlabeled']
            else:
                self.labeled_batch_sampler, self.unlabeled_batch_sampler=copy.copy(batch_sampler),copy.copy(batch_sampler)
            if mu is not None:
                if hasattr(self.labeled_batch_sampler, 'batch_size') and hasattr(self.unlabeled_batch_sampler, 'batch_size'):
                    self.unlabeled_batch_sampler.batch_size = self.labeled_batch_sampler.batch_size*mu

        if isinstance(self.labeled_sampler,SemiSampler):
            self.labeled_sampler=self.labeled_sampler.init_sampler(self.labeled_dataset)

        if isinstance(self.labeled_batch_sampler,SemiBatchSampler):
            self.labeled_batch_sampler=self.labeled_batch_sampler.init_sampler(self.labeled_sampler)

        if isinstance(self.unlabeled_sampler,SemiSampler):
            self.unlabeled_sampler=self.unlabeled_sampler.init_sampler(self.unlabeled_dataset)

        if isinstance(self.unlabeled_batch_sampler,SemiBatchSampler):
            self.unlabeled_batch_sampler=self.unlabeled_batch_sampler.init_sampler(self.unlabeled_sampler)

        if self.labeled_batch_sampler is None and self.labeled_sampler is None:
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
        elif self.labeled_batch_sampler is not None:
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

        if self.unlabeled_batch_sampler is None and self.unlabeled_sampler is None:

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

        elif self.unlabeled_batch_sampler is not None:
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
        return self.labeled_dataloader,self.unlabeled_dataloader

