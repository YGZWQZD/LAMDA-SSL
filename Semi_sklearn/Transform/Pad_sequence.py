from Semi_sklearn.Transform.Transformer import Transformer
class Pad_sequence(Transformer):
    def __init__(self,length,pad_val=None):
        super(Pad_sequence, self).__init__()
        self.pad_val=pad_val
        self.length=length
    def transform(self,X):
        if self.pad_val is None:
            if isinstance(X[0],str):
                pad_val="<pad>"
            else:
                pad_val=type(X[0])(0)
        else:
            pad_val=self.pad_val
        cur_length=len(X)
        pad_length=int(max(0,self.length-cur_length))
        if isinstance(X,list):
            X=list(X)
        for _ in range(pad_length):
            X.append(pad_val)
        return X