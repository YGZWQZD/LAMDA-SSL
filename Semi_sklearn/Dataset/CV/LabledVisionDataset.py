from Semi_sklearn.Dataset.LabledDataset import LabledDataset
from torchvision.datasets.vision import VisionDataset
from PIL import Image


class LabledVisionDataset(LabledDataset,VisionDataset):
    def __init__(self,
                 root=None,
                 transforms=None,
                 transform=None,
                 target_transform=None
                 ):
        LabledDataset.__init__(self)
        VisionDataset.__init__(self,root=root,transforms=transforms,transform=transform,target_transform=target_transform)

    def __getitem__(self, i):

        X, y = self.X, self.y

        Xi = X[i]
        yi = y[i] if y is not None else None
        Xi = Image.fromarray(Xi)

        Xi, yi = self._transform(Xi, yi)
        if self.transform is not None:
            Xi = self.transform(Xi)

        if self.target_transform is not None:
            yi = self.target_transform(yi)

        return Xi, yi

    def __len__(self):
        return LabledDataset.__len__(self)


