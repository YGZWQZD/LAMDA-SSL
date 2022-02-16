from Semi_sklearn.Transform.Transformer import Transformer
from torchvision import transforms
class ToTensor(Transformer):
    def __init__(self):
        super().__init__()
    def transform(self,X):
        return transforms.ToTensor()(X)
