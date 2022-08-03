import copy
from LAMDA_SSL.utils import indexing
from torch.utils.data.dataset import Dataset
from LAMDA_SSL.utils import get_len,get_indexing_method

class LabeledDataset(Dataset):
    def __init__(self,
                 transforms=None,
                 transform=None,
                 target_transform=None,
                 pre_transform=None
                 ):
        # >> Parameter
        # >> - pre_transform: The way to preprocess X before augmentation.
        # >> - transforms: The way to transform X and y at the same time after data augmentation.
        # >> - transform: The way to transform X after data augmentation.
        # >> - target_transform: The way to transform y after data augmentation.
        self.transforms=transforms
        self.transform=transform
        self.pre_transform=pre_transform
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
        self.len=get_len(self.X)
        self.X_indexing_method = get_indexing_method(self.X)
        self.y_indexing_method = get_indexing_method(self.y)
        self.data_initialized = True
        return self

    def _transforms(self,X,y,transforms):
        X, y = copy.deepcopy(X), copy.deepcopy(y)
        if isinstance(transforms,(list,tuple)):
            for item in transforms:
                X,y=self._transforms(self.X,y,item)
        elif callable(transforms):
            X,y=transforms(X,y)
        elif hasattr(transforms,'fit_transform'):
            X,y=transforms.fit_transform(X,y)
        elif hasattr(transforms, 'forward'):
            X, y = transforms.forward(X, y)
        elif hasattr(transforms,'transform'):
            X, y = transforms.transform(X, y)
        else:
            raise Exception('Transforms is not Callable!')
        return X,y

    def to_list(self,l):
        if isinstance(l, tuple):
            l= list(l)
        elif not isinstance(l,list):
            l=[l]
        return l

    def insert(self,l,pos,item):
        if isinstance(l,dict):
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

    def add_transforms(self,transforms,dim=1,x=0,y=0):
        if transforms is None and dim == 1:
            return
        if transforms is None and dim == 0 and x==0:
            return
        if self.transforms is None and dim==0 and x==0:
            self.transforms=transforms
        elif dim==0:
            self.transforms=self.insert(self.transforms,x,transforms)
        else:
            if not isinstance(self.transforms, (dict, tuple, list)):
                self.transforms=[self.transforms]
            self.transforms[x]=self.insert(self.transforms[x],y,transforms)

    def add_target_transform(self,target_transform,dim=1,x=0,y=0):
        if target_transform is None and dim == 1:
            return
        if target_transform is None and dim == 0 and x==0:
            return
        if self.target_transform is None and dim==0 and x==0:
            self.target_transform=target_transform
        elif dim==0:
            self.target_transform=self.insert(self.target_transform,x,target_transform)
        else:
            if not isinstance(self.target_transform, (dict, tuple, list)):
                self.target_transform=[self.target_transform]
            self.target_transform[x]=self.insert(self.target_transform[x],y,target_transform)

    def _transform(self,X,transform):
        X = copy.deepcopy(X)
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

    def apply_transform(self,X,y):
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
        if self.transforms is not None:
            if isinstance(self.transforms,(tuple,list)):
                list_X=[]
                list_y=[]
                for item in self.transforms:
                    _X,_y=self._transforms(X,y,item)
                    list_X.append(_X)
                    list_y.append(_y)
                X=list_X if len(list_X) is not 1 else list_X[0]
                y=list_y if len(list_y) is not 1 else list_y[0]
            elif isinstance(self.transforms,dict):
                dict_X={}
                dict_y={}
                for key, val in self.transforms.items():
                    _X,_y=self._transforms(X,y,val)
                    dict_X[key]=_X
                    dict_y[key]=_y
                X = dict_X
                y = dict_y
            else:
                X,y=self._transforms(X,y,self.transforms)
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
                    dict_X[key] = _X
                X = dict_X
            else:
                X=self._transform(X,self.transform)
        if self.target_transform is not None:
            if isinstance(self.target_transform, (tuple, list)):
                list_y = []
                for item in self.target_transform:
                    _y = self._transform(y, item)
                    list_y.append(_y)
                y = list_y
            elif isinstance(self.target_transform, dict):
                dict_y = {}
                for key, val in self.target_transform.items():
                    _y = self._transform(y, val)
                    dict_y[key] = _y
                y = dict_y
            else:
                y=self._transform(y,self.target_transform)
        return X,y

    def __getitem__(self, i):
        X, y = self.X, self.y
        Xi = indexing(X,i)
        yi = indexing(y,i)
        Xi, yi = self.apply_transform(Xi, yi)
        return i,Xi, yi

    def __len__(self):
        return get_len(self.X) if self.len is None else self.len