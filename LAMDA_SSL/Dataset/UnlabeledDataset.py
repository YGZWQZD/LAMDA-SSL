import copy
import torch
from LAMDA_SSL.utils import indexing
from torch.utils.data.dataset import Dataset
from LAMDA_SSL.utils import get_len,get_indexing_method

class UnlabeledDataset(Dataset):
    def __init__(self,
                 pre_transform=None,
                 transform=None
                 ):
        # >> Parameter
        # >> - pre_transform: The way to preprocess X before augmentation.
        # >> - transform: The way to transform X after data augmentation.
        self.transform=transform
        self.pre_transform = pre_transform
        self.X=None
        self.y=None
        self.len=None
        self.X_indexing_method=None
        self.y_indexing_method=None
        self.data_initialized=False

    def init_dataset(self, X=None, y=None):
        self.X = X
        self.y = y
        self.len=get_len(X)
        self.X_indexing_method = get_indexing_method(self.X)
        self.y_indexing_method = get_indexing_method(self.y)
        self.data_initialized = True
        return self

    def to_list(self,l):
        if isinstance(l, tuple):
            l= list(l)
        elif not isinstance(l,list):
            l=[l]
        return l

    def insert(self,l,pos,item):
        if l is None and pos==0:
            l=item
        elif isinstance(l,dict):
            l[pos]=item
        else:
            l=self.to_list(l)
            l = l[:pos] + [item] + l[pos:]
        return l

    def add_transform(self,transform,dim=1,x=0,y=0):
        if transform is None and dim == 1:
            return
        if transform is None and dim == 0 and x==0:
            return
        if self.transform is None and dim==0 and x==0:
            self.transform=transform
        elif dim==0:
            self.transform=self.insert(self.transform,x,transform)
        else:
            if not isinstance(self.transform, (dict, tuple, list)):
                self.transform=[self.transform]
            self.transform[x]=self.insert(self.transform[x],y,transform)

    def add_pre_transform(self,transform,dim=1,x=0,y=0):
        if transform is None and dim == 1:
            return
        if transform is None and dim == 0 and x==0:
            return
        if self.pre_transform is None and dim==0 and x==0:
            self.pre_transform=transform
        elif dim==0:
            self.pre_transform=self.insert(self.pre_transform,x,transform)
        else:
            if not isinstance(self.pre_transform, (dict, tuple, list)):
                self.pre_transform=[self.pre_transform]
            self.pre_transform[x]=self.insert(self.pre_transform[x],y,transform)

    def _transform(self,X,transform):

        if isinstance(transform,(list,tuple)):
            for item in transform:
                X=self._transform(X,item)
        elif callable(transform):
            X = transform(X)
        elif hasattr(transform,'fit_transform'):
            X = transform.fit_transform(X)
        elif hasattr(transform,'transform'):
            X = transform.transform(X)
        elif hasattr(transform,'forward'):
            X = transform.forward(X)
        else:
            raise Exception('Transforms is not Callable!')
        return X

    def apply_transform(self,X,y=None):
        if self.pre_transform is not None:
            if isinstance(self.pre_transform,(tuple,list)):
                list_X=[]
                for item in self.pre_transform:
                    _X=self._transform(X,item)
                    list_X.append(_X)
                X=list_X

            elif isinstance(self.pre_transform,dict):
                dict_X={}
                for key, val in self.pre_transform.items():
                    _X=self._transform(X,val)
                    dict_X[key]=_X
                X = dict_X

            else:
                X=self._transform(X,self.pre_transform)

        if self.transform is not None:

            if isinstance(self.transform, (tuple, list)):
                list_X = []
                for item in self.transform:
                    # print(item)
                    _X = self._transform(X, item)
                    list_X.append(_X)
                X = list_X

            elif isinstance(self.transform, dict):
                dict_X = {}
                for key, val in self.transform.items():
                    _X = self._transform(X, val)
                    dict_X[key] = _X
                X = dict_X
            else:
                X=self._transform(X,self.transform)
        y=torch.Tensor([-1]) if y is None else y
        return X,y

    def __getitem__(self, i):
        X, y = self.X, self.y

        Xi = indexing(X,i)
        yi = indexing(y,i)
        Xi = copy.deepcopy(Xi)
        yi=copy.deepcopy(yi)

        Xi, yi = self.apply_transform(Xi, yi)
        return i,Xi, yi

    def __len__(self):
        return get_len(self.X) if self.len is None else self.len