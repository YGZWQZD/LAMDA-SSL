import copy
from distutils.version import LooseVersion
from functools import partial
from numbers import Number
from reprlib import recursive_repr
import sklearn
import torch
from torch.utils.data import dataset
import numpy as np
from scipy import sparse
from torch.nn.utils.rnn import PackedSequence
import torch.nn as nn


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

def indexing_sparse(data,i):
    data=copy.copy(data)
    data = data.toarray().squeeze(0)
    return data[i]

def indexing_ndframe(data, i):
    if hasattr(data, 'iloc'):
        data=data.copy(data)
        data = {k: data[k].values.reshape(-1, 1) for k in data}
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
        return indexing_dict

    if is_sparse(data):
        return indexing_sparse

    if isinstance(data, (list, tuple)):
        try:
            if isinstance(data[0],(Number,str)):
                raise TypeError('Can not index data!')
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

def _len(data):
    if isinstance(data,(Number,str)):
        raise TypeError('Can not get the lengeth of data!')
    if data is None:
        return 0
    elif isinstance(data,torch.utils.data.Dataset):
        return data.__len__()
    elif is_sparse(data):
        return data.shape[0]
    else:
        return len(data)

def get_len(data):
    lens = [apply_to_data(data, _len, unpack_dict=True)]
    lens = list(flatten(lens))
    len_set = set(lens)
    if len(len_set) != 1:
        raise ValueError("Dataset does not have consistent lengths.")
    return list(len_set)[0]

def is_torch_data_type(x):
    # pylint: disable=protected-access
    return isinstance(x, (torch.Tensor, PackedSequence))

def to_device(X, device):

    if device is None:
        return X

    if isinstance(X, dict):
        return {key: to_device(val, device) for key, val in X.items()}

    if isinstance(X, (tuple, list)) and (type(X) != PackedSequence):
        return type(X)(to_device(x, device) for x in X)

    if isinstance(X, torch.distributions.distribution.Distribution):
        return X

    return X.to(device)

# def to_tensor(X, device=None, accept_sparse=False):
#     to_tensor_ = partial(to_tensor, device=device)
#     if is_torch_data_type(X):
#         return to_device(X, device)
#     if isinstance(X, dict):
#         return {key: to_tensor_(val) for key, val in X.items()}
#     if isinstance(X, (list, tuple)):
#         try:
#             indexing(X[0],0)
#             return [to_tensor_(x) for x in X]
#         except:
#             return torch.as_tensor(np.array(X), device=device)
#     if np.isscalar(X):
#         return torch.as_tensor(X, device=device)
#     if isinstance(X, Sequence):
#         return torch.as_tensor(np.array(X), device=device)
#     if isinstance(X, np.ndarray):
#         return torch.as_tensor(X, device=device)
#     if sparse.issparse(X):
#         if accept_sparse:
#             return torch.sparse_coo_tensor(
#                 X.nonzero(), X.data, size=X.shape).to(device)
#         raise TypeError("Sparse matrices are not supported. Set "
#                         "accept_sparse=True to allow sparse matrices.")
#
#     raise TypeError("Cannot convert this data type to a torch tensor.")


def to_numpy(X):
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, dict):
        return np.array(to_numpy(val) for key, val in X.items())
    if is_pandas_ndframe(X):
        return X.values
    if isinstance(X, (tuple, list)):
        return np.array(X)
    if not is_torch_data_type(X):
        raise TypeError("Cannot convert this data type to a numpy array.")
    if X.is_cuda:
        X = X.cpu()
    if X.requires_grad:
        X = X.detach()
    return X.numpy()

# def to_image(X):
#     if isinstance(X,Image.Image):
#         return X
#     else:
#         X=to_numpy(X)
#         X=Image.fromarray(X)
#         return X

class partial:
    """New function with partial application of the given arguments
    and keywords.
    """
    __slots__ = "func", "args", "keywords", "__dict__", "__weakref__"

    def __new__(*args, **keywords):
        if not args:
            raise TypeError("descriptor '__new__' of partial needs an argument")
        if len(args) < 2:
            raise TypeError("type 'partial' takes at least one argument")
        cls, func, *args = args
        if not callable(func):
            raise TypeError("the first argument must be callable")
        args = tuple(args)

        if hasattr(func, "func"):
            args = func.args + args
            tmpkw = func.keywords.copy()
            tmpkw.update(keywords)
            keywords = tmpkw
            del tmpkw
            func = func.func

        self = super(partial, cls).__new__(cls)

        self.func = func
        self.args = args
        self.keywords = keywords
        return self

    def __call__(*args, **keywords):
        if not args:
            raise TypeError("descriptor '__call__' of partial needs an argument")
        self, *args = args
        newkeywords = self.keywords.copy()
        newkeywords.update(keywords)
        return self.func(*self.args, *args, **newkeywords)
    def change(self,**keywords):
        self.keywords.update(keywords)
        return self
    @recursive_repr()
    def __repr__(self):
        qualname = type(self).__qualname__
        args = [repr(self.func)]
        args.extend(repr(x) for x in self.args)
        args.extend(f"{k}={v!r}" for (k, v) in self.keywords.items())
        if type(self).__module__ == "functools":
            return f"functools.{qualname}({', '.join(args)})"
        return f"{qualname}({', '.join(args)})"

    def __reduce__(self):
        return type(self), (self.func,), (self.func, self.args,
               self.keywords or None, self.__dict__ or None)

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 4 items in state, got {len(state)}")
        func, args, kwds, namespace = state
        if (not callable(func) or not isinstance(args, tuple) or
           (kwds is not None and not isinstance(kwds, dict)) or
           (namespace is not None and not isinstance(namespace, dict))):
            raise TypeError("invalid partial state")

        args = tuple(args) # just in case it's a subclass
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict: # XXX does it need to be *exactly* dict?
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.func = func
        self.args = args
        self.keywords = kwds

class EMA:
    """
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class class_status:
    def __init__(self,y):
        self.y=y
        try:
            self.y_arr = to_numpy(self.y)
        except (AttributeError, TypeError):
            self.y_arr = self.y
        if self.y_arr.ndim == 2:
            self.y_arr = np.array([" ".join(row.astype("str")) for row in self.y_arr])
    @property
    def classes(self):
        classes, y_indices = np.unique(self.y_arr, return_inverse=True)
        return classes

    @property
    def y_indices(self):
        classes, y_indices = np.unique(self.y_arr, return_inverse=True)
        return y_indices

    @property
    def num_classes(self):
        classes, y_indices = np.unique(self.y_arr, return_inverse=True)
        num_class = classes.shape[0]
        return num_class

    @property
    def class_counts(self):
        classes, y_indices = np.unique(self.y_arr, return_inverse=True)
        class_counts = np.bincount(y_indices)
        return class_counts


def _l2_normalize(d):
    d /= (torch.sqrt(torch.sum(d ** 2, dim=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    return d


def one_hot(targets, nClass,device):
    logits = torch.zeros(targets.size(0), nClass).to(device)
    return logits.scatter_(1, targets.unsqueeze(1).long(), 1)

class Bn_Controller:
    def __init__(self):
        """
        freeze_bn and unfreeze_bn must appear in pairs
        """
        self.backup = {}

    def freeze_bn(self, model):
        assert self.backup == {}
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                self.backup[name + '.running_mean'] = m.running_mean.data.clone()
                self.backup[name + '.running_var'] = m.running_var.data.clone()
                self.backup[name + '.num_batches_tracked'] = m.num_batches_tracked.data.clone()

    def unfreeze_bn(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.backup[name + '.running_mean']
                m.running_var.data = self.backup[name + '.running_var']
                m.num_batches_tracked.data = self.backup[name + '.num_batches_tracked']
        self.backup = {}
