import PIL.Image
import torch

from Semi_sklearn.Transform.Transformer import Transformer
from torchvision import transforms
class ToTensor(Transformer):
    def __init__(self):
        super().__init__()

    def transform(self,X):
        if isinstance(X,PIL.Image.Image):
            return transforms.ToTensor()(X)
        else:
            return torch.Tensor(X)