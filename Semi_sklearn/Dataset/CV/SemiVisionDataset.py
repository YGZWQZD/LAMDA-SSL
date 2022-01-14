from torchvision.datasets.vision import VisionDataset
from Semi_sklearn.Dataset.SemiDataset import SemiDataset
from PIL import Image
from Semi_sklearn.Dataset.CV.LabledVisionDataset import LabledVisionDataset
from Semi_sklearn.Dataset.CV.UnlabledVisionDataset import UnlabledVisionDataset
from Semi_sklearn.Dataset.CV.SemiTrainVisionDataset import SemiTrainVisionDataset
from Semi_sklearn.utils import indexing
from Semi_sklearn.utils import partial
class SemiVisionDataset(SemiDataset,VisionDataset):
    def __init__(
        self,
        root=None,
        transforms= None,
        transform= None,
        target_transform= None,
        test_size=None,
        labled_size=0.1,
        stratified=False,
        shuffle=True,
        random_state=None
    ) -> None:
        print(2)
        self.root=root
        self.transform=transform
        self.transforms=transforms
        self.target_transform=target_transform
        self.labled_size=labled_size
        self.test_size=test_size
        self.stratified=stratified
        self.shuffle=shuffle
        self.random_state=random_state
        self.labled_X=None
        self.labled_y=None
        self.unlabled_X=None
        self.unlabled_y=None
        self.test_X=None
        self.test_y=None
        self.labled_dataset=None
        self.unlabled_dataset=None
        self.train_dataset=None
        self.test_dataset=None
        self.len_test=None
        self.len_labled=None
        self.len_unlabled=None
        self.data_initialized = False
        SemiDataset.__init__(self,test_size=test_size,labled_size=labled_size,stratified=stratified,shuffle=shuffle,random_state=random_state)
        VisionDataset.__init__(self,root=root,transforms=transforms,transform=transform,target_transform=target_transform)
        self.labled_class=partial(LabledVisionDataset,root=self.root,transforms=self.transforms,
                                  transform=self.transform,target_transform=self.target_transform)
        self.unLabled_class=partial(UnlabledVisionDataset,root=self.root,transforms=self.transforms,
                                  transform=self.transform,target_transform=self.target_transform)
        self.semitrain_class=partial(SemiTrainVisionDataset,root=self.root,transforms=self.transforms,
                                  transform=self.transform,target_transform=self.target_transform,
                                  labled_size=self.labled_size,stratified=self.stratified,
                                  shuffle=self.shuffle,random_state=self.random_state)

    def _init_dataset(self):
        raise NotImplementedError(
            "_init_dataset method of SemiVisionDataset class must be implemented."
        )

    def __getitem__(self, i, train=False, labled=True):
        if train is not True:
            X, y = self.test_X, self.test_y
            X_indexing_method=self.test_X_indexing_method
            y_indexing_method=self.test_y_indexing_method
        elif labled:
            X, y = self.labled_X, self.labled_y
            X_indexing_method = self.labled_X_indexing_method
            y_indexing_method = self.labled_y_indexing_method
        else:
            X, y = self.unlabled_X, self.unlabled_y
            X_indexing_method = self.unlabled_X_indexing_method
            y_indexing_method = self.unlabled_y_indexing_method

        Xi = indexing(X, i, X_indexing_method)
        yi = indexing(y, i, y_indexing_method)
        # Xi = Image.fromarray(Xi)

        Xi,yi=self._transform(Xi, yi)
        if self.transform is not None:
            Xi=self.transform(Xi)
            # if isinstance(self.transform,(list,tuple)):
            #     trans_Xi=[trans(Xi) for trans in self.transform]
            # elif isinstance(self.transform,dict):
            #     trans_Xi={}
            #     for key,val in self.transform:
            #         trans_Xi[key]=val(Xi)
            # else:
            #     trans_Xi=self.transform(Xi)
            # Xi = trans_Xi

        if self.target_transform is not None:
            yi = self.target_transform(yi)


        return Xi,yi

    def __len__(self, train=False, labled=True):
        return SemiDataset.__len__(self,train, labled)











    