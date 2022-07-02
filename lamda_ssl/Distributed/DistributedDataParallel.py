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