from lamda_ssl.Transform.Transformer import Transformer
from lamda_ssl.Transform.Pad_sequence import Pad_sequence
from lamda_ssl.Transform.Truncate import Truncate
class Adjust_length(Transformer):
    def __init__(self,length,pad_val=None,pos=0):
        super().__init__()
        self.length=length
        self.pad=Pad_sequence(self.length,pad_val)
        self.truncate=Truncate(length,pos)
    def transform(self,X):
        if len(X)<self.length:
            X=self.pad(X)
        else:
            X=self.truncate(X)
        # print('adjust')
        # print(len(X))
        return X
