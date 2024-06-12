from LAMDA_SSL.Base.Transformer import Transformer
from LAMDA_SSL.Transform.Text.PadSequence import PadSequence
from LAMDA_SSL.Transform.Text.Truncate import Truncate
import jieba

class Lcut(Transformer):
    def __init__(self):
        super().__init__()

    def transform(self,X):
        X = jieba.lcut(X)
        return X
