from torchvision.datasets import VisionDataset
from Semi_sklearn.dataset import SemiDataset
from skorch.utils import check_indexing,multi_indexing,is_pandas_ndframe
from skorch.dataset import get_len
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple
class SemiVisionDataset(VisionDataset):
    def __init__(
        self,
        root=None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        labled_X=None,
        labled_y=None,
        unlabled_X=None,
        unlabled_y=None,
        test_X=None,
        test_y=None,
        labled_size=0.1,
        test_size=None,
        stratified=False,
        shuffle=True,
        random_state=None
    ) -> None:
        self.labled_X=None
        self.labled_y=None
        self.unlabled_X=None
        self.unlabled_y=None
        self.test_X=None
        self.test_y=None
        self.labled_X_indexing = check_indexing(labled_X)
        self.labled_y_indexing = check_indexing(labled_y)
        self.labled_size=labled_size
        self.test_size=test_size
        self.stratified=stratified
        self.shuffle=shuffle
        self.random_state=random_state
        if root is not None:
            super().__init__(self,root,transforms,transform,target_transform)
            self.labled_X,self.labled_y=self.get_data(test=False,labled=True)
            self.unlabled_X,self.unlabled_y=self.get_data(test=False,labled=False)
            self.test_X,test_y=self.get_data(test=True,labled=True)
            
        elif labled_X is not None:
            # self.labled_X_is_ndframe = is_pandas_ndframe(labled_X)
            self.root=None
            self.transform=transform
            self.target_transform=target_transform
            self.transforms=transforms
            if test_X is not None:
                self.test_X=test_X
                self.test_y=test_y
            elif test_size is not None:
                self.test_X,self.test_y,self.labled_X,self.labled_y=SemiSplit(X=self.labled_X,y=self.labled_y,
                                                                labled_size=self.test_size,
                                                                stratified=self.stratified,
                                                                shuffle=self.shuffle,
                                                                random_state=self.random_state,
                                                                X_indexing=self.labled_X_indexing, 
                                                                y_indexing=self.labled_y_indexing
                                                                )
                
            else:
                raise ValueError("Can't get test dataset") 
            if unlabled_X is not None:
                self.unlabled_X=unlabled_X
                self.unlabled_y=unlabled_y
            elif test_size is not None:
                self.labled_X,self.labled_y,self.unlabled_X,self.unlabled_y=SemiSplit(X=self.labled_X,y=self.labled_y,
                                                                labled_size=self.labled_size,
                                                                stratified=self.stratified,
                                                                shuffle=self.shuffle,
                                                                random_state=self.random_state,
                                                                X_indexing=self.labled_X_indexing, 
                                                                y_indexing=self.labled_y_indexing
                                                                )
            else:
                raise ValueError("Can't get unlabled dataset")         
        else:
            raise ValueError("Can't get dataset")
        self._labled_len=get_len(self.labled_X)
        self.labled_X_indexing = check_indexing(self.labled_X) if self.labled_X_indexing is None else self.labled_X_indexing
        self.labled_y_indexing = check_indexing(self.labled_y) if self.labled_y_indexing is None else self.labled_y_indexing
        # self.labled_X_is_ndframe = is_pandas_ndframe(self.labled_X) if self.labled_X_is_ndframe is None else self.labled_X_is_ndframe
        self._test_len=get_len(self.test_X)
        # self.test_X_is_ndframe = is_pandas_ndframe(self.test_X)
        self.test_X_indexing=check_indexing(self.test_X)
        self.test_y_indexing=check_indexing(self.test_y)
        self._unlabled_len=get_len(self.unlabled_X)
        self.unlabled_X_indexing = check_indexing(self.unlabled_X)
        self.unlabled_y_indexing = check_indexing(self.unlabled_y)
        # self.unlabled_X_is_ndframe = is_pandas_ndframe(self.unlabled_X)
    # def get_dataset(self,root,transforms,transform,target_transform,labled_size,test_size,test):
    def get_data(self,test,labled):
        return None
    def __len__(self,test=False,labled = True):
        if test:
            return self._test_len
        if labled:
            return self._labled_len
        else:
            return self._unlabled_len

    def transform_(self, X, y, test=False,labled = True):
        if test or labled:
            y = torch.Tensor([0]) if y is None else y
            if sparse.issparse(X):
                X = X.toarray().squeeze(0)
            X=self.transform(X)
            y=self.target_transform(y)
            return X, y
        else:
            if sparse.issparse(X):
                X = X.toarray().squeeze(0)
            X=self.transform(X)
            return X
    def __getitem__(self, i, test=False, labled = True):
        if test:
            X, y = self.test_X, self.test_y
            # if self.test_X_is_ndframe:
            #     X = {k: X[k].values.reshape(-1, 1) for k in X}
            Xi = multi_indexing(X, i, self.test_X_indexing)
            yi = multi_indexing(y, i, self.test_y_indexing)
            return self.transform_(Xi, yi, test,labled)
        elif labled:
            X, y = self.labled_X, self.labled_y
            # if self.labled_X_is_ndframe:
            #     X = {k: X[k].values.reshape(-1, 1) for k in X}

            Xi = multi_indexing(X, i, self.labled_X_indexing)
            yi = multi_indexing(y, i, self.labled_y_indexing)
            return self.transform_(Xi, yi, test,labled)
        else:
            X, y = self.unlabled_X, self.unlabled_y
            # if self.unlabled_X_is_ndframe:
            #     X = {k: X[k].values.reshape(-1, 1) for k in X}

            Xi = multi_indexing(X, i, self.unlabled_X_indexing)
            yi = multi_indexing(y, i, self.unlabled_y_indexing)
            return self.transform_(Xi, yi,test,labled)









    