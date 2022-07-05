from torch.nn import parallel
class DistributedDataParallel:
    def __init__(
        self,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        gradient_as_bucket_view=False,
    ):
        # >> Parameter
        # >> - device_ids: Available GPUs.
        # >> - output_device: The GPU where the output result is stored.
        # >> - dim: The dimension of data aggregation from each device.
        # >> - broadcast_buffers: Flag that enables syncing (broadcasting) buffers of the module at beginning of the 'forward' function.
        # >> - process_group: The process group to be used for distributed data all-reduction. If None, the default process group, which is created by :func:'torch.distributed.init_process_group', will be used.
        # >> - bucket_cap_mb: 'DistributedDataParallel' will bucket parameters into multiple buckets so that gradient reduction of each bucket can potentially overlap with backward computation. :attr:'bucket_cap_mb' controls the bucket size in MegaBytes (MB).
        # >> - find_unused_parameters: Traverse the autograd graph from all tensors contained in the return value of the wrapped module's 'forward' function. Parameters that don't receive gradients as part of this graph are preemptively marked as being ready to be reduced. In addition, parameters that may have been used in the wrapped module's 'forward' function but were not part of loss computation and thus would also not receive gradients are preemptively marked as ready to be reduced.
        # >> - gradient_as_bucket_view: When set to True, gradients will be views pointing to different offsets of 'allreduce' communication buckets. This can reduce peak memory usage, where the saved memory size will be equal to the total gradients size. Moreover, it avoids the overhead of copying between gradients and 'allreduce' communication buckets. When gradients are views, detach_() cannot be called on the gradients. If hitting such errors, please fix it by referring to the :meth: '~torch.optim.Optimizer.zero_grad' function in 'torch/optim/optimizer.py' as a solution.
        self.device_ids=device_ids
        self.output_device=output_device
        self.dim=dim
        self.broadcast_buffers=broadcast_buffers
        self.process_group=process_group
        self.bucket_cap_mb=bucket_cap_mb
        self.find_unused_parameters=find_unused_parameters
        self.gradient_as_bucket_view=gradient_as_bucket_view
    def init_parallel(self,module):
        return parallel.DistributedDataParallel(module=module,
                                                device_ids=self.device_ids,
                                                output_device=self.output_device,
                                                dim=self.dim,
                                                broadcast_buffers=self.broadcast_buffers,
                                                process_group=self.process_group,
                                                bucket_cap_mb=self.bucket_cap_mb,
                                                find_unused_parameters=self.find_unused_parameters,
                                                gradient_as_bucket_view=self.gradient_as_bucket_view)