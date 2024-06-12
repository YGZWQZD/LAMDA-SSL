from LAMDA_SSL.Base.Transformer import Transformer
class Truncate(Transformer):
    def __init__(self,length=300,pos=0):
        # >> Paraameter:
        # >> - length: The length of the truncated text .
        # >> - pos: The position to start truncating.
        super().__init__()
        self.length=length
        self.pos=pos

    def transform(self,X):

        assert len(X)>=self.pos+self.length

        return X[self.pos:self.pos+self.length]

