from torchvision.datasets.vision import VisionDataset
from Semi_sklearn.Dataset.SemiTrainDataset import SemiTrainDataset
from Semi_sklearn.Dataset.CV.LabledVisionDataset import LabledVisionDataset
from Semi_sklearn.Dataset.CV.UnlabledVisionDataset import UnlabledVisionDataset
from Semi_sklearn.utils import indexing
from Semi_sklearn.utils import partial

class SemiTrainVisionDataset(SemiTrainDataset, VisionDataset):
    def __init__(
            self,
            root=None,
            transforms=None,
            transform=None,
            target_transform=None,
            labled_size=0.1,
            stratified=False,
            shuffle=True,
            random_state=None
    ) -> None:
        self.root = root
        self.transform = transform
        self.transforms = transforms
        self.target_transform = target_transform
        self.labled_size = labled_size
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        self.labled_X = None
        self.labled_y = None
        self.unlabled_X = None
        self.unlabled_y = None
        self.labled_dataset = None
        self.unlabled_dataset = None
        self.train_dataset = None
        self.len_labled = None
        self.len_unlabled = None
        self.data_initialized = False

        SemiTrainDataset.__init__(self, labled_size=labled_size, stratified=stratified, shuffle=shuffle,
                             random_state=random_state)
        VisionDataset.__init__(self, root=root, transforms=transforms, transform=transform,
                               target_transform=target_transform)

        self.labled_class=partial(LabledVisionDataset,root=self.root,transforms=self.transforms,
                                  transform=self.transform,target_transform=self.target_transform)
        self.unlabled_class = partial(UnlabledVisionDataset, root=self.root, transforms=self.transforms,
                                    transform=self.transform, target_transform=self.target_transform)
    def _init_dataset(self):
        raise NotImplementedError(
            "_init_dataset method of SemiVisionDataset class must be implemented."
        )

    def __getitem__(self, i, labled=True):
        if labled:
            X, y = self.labled_X, self.labled_y
            X_indexing_method=self.labled_X_indexing_method
            y_indexing_method=self.labled_y_indexing_method
        else:
            X, y = self.unlabled_X, self.unlabled_y
            X_indexing_method = self.unlabled_X_indexing_method
            y_indexing_method = self.unlabled_y_indexing_method

        Xi = indexing(X, i, X_indexing_method)
        yi = indexing(y, i, y_indexing_method)
        # Xi = Image.fromarray(Xi)

        Xi, yi = self._transform(Xi, yi)

        if self.transform is not None:
            Xi=self.transform(Xi)

        if self.target_transform is not None:
            yi = self.target_transform(yi)

        return Xi, yi

    def __len__(self,  labled=True):
        return SemiTrainDataset.__len__(self, labled)











