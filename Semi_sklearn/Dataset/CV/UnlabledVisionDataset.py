from Semi_sklearn.Dataset.UnlabledDataset import UnlabledDataset
from torchvision.datasets.vision import VisionDataset
from PIL import Image


class UnlabledVisionDataset(UnlabledDataset,VisionDataset):
    def __init__(self,
                 root=None,
                 transforms=None,
                 transform=None,
                 target_transform=None
                 ):
        UnlabledDataset.__init__(self)
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
        return UnlabledDataset.__len__(self)


