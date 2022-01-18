from torch import nn
class DataParallel:
    def __init__(self, device_ids=None, output_device=None, dim=0):
        self.device_ids=device_ids
        self.output_device=output_device
        self.dim=dim
    def init_parallel(self,module):
        return nn.DataParallel(module=module,device_ids=self.device_ids,
                               output_device=self.output_device,dim=self.dim)
