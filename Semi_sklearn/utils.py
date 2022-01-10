# pylint: disable=unused-argument
from copy import deepcopy
from distutils.version import LooseVersion
from functools import partial
import sklearn
import torch
from torch.utils.data import dataset
import numpy as np
from scipy import sparse


if LooseVersion(sklearn.__version__) >= '0.22.0':
    from sklearn.utils import _safe_indexing as safe_indexing
else:
    from sklearn.utils import safe_indexing

def is_pandas_ndframe(x):
    return hasattr(x, 'iloc')

def indexing_none(data, i):
    return None


def indexing_dict(data, i):
    return {k: v[i] for k, v in data.items()}


def indexing_list_tuple_of_data(data, i, indexings=None):
    if not indexings:
        return [indexing(x, i) for x in data]
    return [indexing(x, i, ind)
            for x, ind in zip(data, indexings)]


def indexing_ndframe(data, i):
    # During fit, DataFrames are converted to dict, which is why we
    # might need _indexing_dict.
    if hasattr(data, 'iloc'):
        return data.iloc[i]
    return indexing_dict(data, i)


def indexing_other(data, i):
    if isinstance(i, (int, np.integer, slice, tuple)):
        return data[i]
    return safe_indexing(data, i)

def indexing_dataset(data,i):
    return data[i]

def get_indexing_method(data):
    if data is None:
        return indexing_none
    if isinstance(data, torch.utils.data.Dataset):
        return indexing_dataset

    if isinstance(data, dict):
        # dictionary of containers
        return indexing_dict

    if isinstance(data, (list, tuple)):
        try:
            indexing(data[0], 0)
            indexings = [get_indexing_method(x) for x in data]
            return partial(indexing_list_tuple_of_data, indexings=indexings)
        except TypeError:

            return indexing_other

    if is_pandas_ndframe(data):

        return indexing_ndframe

    return indexing_other


def normalize_numpy_indices(i):
    if isinstance(i, np.ndarray):
        if i.dtype == bool:
            i = tuple(j.tolist() for j in i.nonzero())
        elif i.dtype == int:
            i = i.tolist()
    return i


def indexing(data, i, indexing_method=None):

    i = normalize_numpy_indices(i)

    if indexing_method is not None:
        return indexing_method(data, i)

    return get_indexing_method(data)(data, i)

def flatten(arr):
    for item in arr:
        if isinstance(item, (tuple, list, dict)):
            yield from flatten(item)
        else:
            yield item

def apply_to_data(data, func, unpack_dict=False):

    apply_ = partial(apply_to_data, func=func, unpack_dict=unpack_dict)

    if isinstance(data, dict):
        if unpack_dict:
            return [apply_(v) for v in data.values()]
        return {k: apply_(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        try:
            return [apply_(x) for x in data]
        except TypeError:
            return func(data)

    return func(data)

def is_sparse(x):
    try:
        return sparse.issparse(x) or x.is_sparse
    except AttributeError:
        return False
def _len(x):
    if is_sparse(x):
        return x.shape[0]
    return len(x)

def get_len(data):
    if isinstance(data,torch.utils.data.Dataset):
        return data.__len__()
    lens = [apply_to_data(data, _len, unpack_dict=True)]
    lens = list(flatten(lens))
    len_set = set(lens)
    if len(len_set) != 1:
        raise ValueError("Dataset does not have consistent lengths.")
    return list(len_set)[0]
class ModelEMA(object):
    def __init__(self, decay):

        self.decay = decay
        self.model=None
        self.device=None
        self.ema=None
    def init_ema(self,model,device):
        self.device=device
        self.model=model
        self.ema = deepcopy(model)
        self.ema.to(device)
        self.ema.eval()
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)


    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])



