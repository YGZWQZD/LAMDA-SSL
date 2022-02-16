import copy
from Semi_sklearn.exceptions import TransformError
from Semi_sklearn.utils import indexing
from torch.utils.data.dataset import Dataset
import os
from Semi_sklearn.utils import get_len,get_indexing_method

class LabledDataset(Dataset):
    def __init__(self,
                 root=None,
                 transforms=None,
                 transform=None,
                 target_transform=None
                 ):
        if isinstance(root, (str, bytes)):
            root = os.path.expanduser(root)
        self.root = root
        self.transforms=transforms
        self.transform=transform
        self.target_transform=target_transform
        self.X=None
        self.y=None
        self.len=None
        self.X_indexing_method=None
        self.y_indexing_method=None
        self.data_initialized=False

    def init_dataset(self, X=None, y=None):
        self.X = X
        self.y = y
        self.X_indexing_method = get_indexing_method(self.X)
        self.y_indexing_method = get_indexing_method(self.y)
        self.data_initialized = True

    def _transforms(self,X,y,transforms):
        X, y = copy.deepcopy(X), copy.deepcopy(y)
        if isinstance(transforms,(list,tuple)):
            for item in transforms:
                X,y=self._transforms(self.X,y,item)
        elif callable(transforms):
            X,y=transforms(X,y)
        elif hasattr(transforms,'fit_transform'):
            X,y=transforms.fit_transform(X,y)
        elif hasattr(transforms,'transform'):
            X, y = transforms.transform(X, y)
        elif hasattr(transforms,'forward'):
            X, y = transforms.forward(X,y)
        else:
            raise TransformError('Transforms is not Callable!')
        return X,y

    def _transform(self,X,transform):
        X = copy.deepcopy(X)
        if isinstance(transform,(list,tuple)):
            for item in transform:
                X=self._transform(self.X,item)
        elif callable(transform):
            X = transform(X)
        elif hasattr(transform,'fit_transform'):
            X = transform.fit_transform(X)
        elif hasattr(transform,'transform'):
            X = transform.transform(X)
        elif hasattr(transform,'forward'):
            X = transform.forward(X)
        else:
            raise TransformError('Transforms is not Callable!')
        return X

    def apply_transform(self,X,y):
        if self.transforms is not None:
            if isinstance(self.transforms,(tuple,list)):
                list_X=[],list_y=[]
                for item in self.transforms:
                    _X,_y=self._transforms(X,y,item)
                    list_X.append(_X)
                    list_y.append(_y)
                X=list_X
                y=list_y
            elif isinstance(self.transforms,dict):
                dict_X={}
                dict_y={}
                for key, val in self.transforms.items():
                    _X,_y=self._transforms(X,y,val)
                    dict_X[key]=val
                    dict_y[key]=val
                X = dict_X
                y = dict_y
            else:
                X,y=self._transforms(X,y,self.transforms)
        else:
            if self.transform is not None:
                if isinstance(self.transform, (tuple, list)):
                    list_X = []
                    for item in self.transform:
                        _X = self._transform(X, item)
                        list_X.append(_X)
                    X = list_X
                elif isinstance(self.transform, dict):
                    dict_X = {}
                    for key, val in self.transform.items():
                        _X = self._transform(X, val)
                        dict_X[key] = val
                    X = dict_X
                else:
                    X=self._transform(X,self.transform)
            if self.target_transform is not None:
                if isinstance(self.target_transform, (tuple, list)):
                    list_y = []
                    for item in self.target_transform:
                        _y, = self._transform(y, item)
                        list_y.append(_y)
                    y = list_y
                elif isinstance(self.target_transform, dict):
                    dict_y = {}
                    for key, val in self.target_transform.items():
                        _y = self._transform(y, val)
                        dict_y[key] = val
                    y = dict_y
                else:
                    y=self.target_transform(y,self.target_transform)
        return X,y

    def __getitem__(self, i):
        X, y = self.X, self.y
        Xi = indexing(X,i)
        yi = indexing(y,i)
        Xi, yi = self.apply_transform(Xi, yi)
        return i,Xi, yi

    def __len__(self):
        return get_len(self.X)


