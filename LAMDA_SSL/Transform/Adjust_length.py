from LAMDA_SSL.Transform.Transformer import Transformer
from LAMDA_SSL.Transform.Pad_sequence import Pad_sequence
from LAMDA_SSL.Transform.Truncate import Truncate
class Adjust_length(Transformer):
    def __init__(self,length=300,pad_val=None,pos=0):
        # >> Parameter:
        # >> - length: Length of adjusted sentence.
        # >> - pad_val: The padding value for insufficient length of text.
        # >> - pos；If the sentence is too long and needs to be cut, this parameter specifies the position to start cutting.
        super().__init__()
        self.length=length
        self.pad=Pad_sequence(self.length,pad_val)
        self.truncate=Truncate(length,pos)
    def transform(self,X):
        if len(X)<self.length:
            X=self.pad(X)
        else:
            X=self.truncate(X)
        return X
