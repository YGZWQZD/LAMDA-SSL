from LAMDA_SSL.Base.Transformer import Transformer
from LAMDA_SSL.Transform.Text.PadSequence import PadSequence
from LAMDA_SSL.Transform.Text.Truncate import Truncate
class AdjustLength(Transformer):
    def __init__(self,length=300,pad_val=None,pos=0):
        # >> Parameter:
        # >> - length: Length of adjusted sentence.
        # >> - pad_val: The padding value for insufficient length of text.
        # >> - pos；If the sentence is too long and needs to be cut, this parameter specifies the position to start cutting.
        super().__init__()
        self.length=length
        self.pad=PadSequence(self.length,pad_val)
        self.truncate=Truncate(length,pos)

    def transform(self,X):
        if len(X)<self.length:
            X=self.pad(X)
        else:
            X=self.truncate(X)
        return X