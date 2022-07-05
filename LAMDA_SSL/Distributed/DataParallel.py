from torch import nn
class DataParallel:
    def __init__(self, device_ids=None, output_device=None, dim=0):
        # >> Parameter
        # >> - device_ids: Available GPUs.
        # >> - output_device: The GPU where the output result is stored.
        # >> - dim: The dimension of data aggregation from each device.
        self.device_ids=device_ids
        self.output_device=output_device
        self.dim=dim
    def init_parallel(self,module):
        return nn.DataParallel(module=module,device_ids=self.device_ids,
                               output_device=self.output_device,dim=self.dim)
