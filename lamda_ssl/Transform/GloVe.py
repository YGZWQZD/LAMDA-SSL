from torchtext import vocab
from lamda_ssl.Transform.Transformer import Transformer
import torch
class Glove(Transformer):
    def __init__(self,name="840B", dim=300,lower_case_backup=True,unk_init=None,pad_init=None,pad_token='<pad>',unk_token='<unk>'):
        super().__init__()
        self.vec=vocab.GloVe(name,dim)
        self.unk_init = torch.Tensor.zero_ if unk_init is None else unk_init
        self.pad_init = torch.Tensor.zero_ if pad_init is None else pad_init
        self.lower_case_backup=lower_case_backup
        self.vec.stoi[pad_token]=self.vec.vectors.shape[0]
        self.vec.stoi[unk_token] = self.vec.vectors.shape[0]+1
        self.vec.vectors=torch.cat([self.vec.vectors,self.pad_init(torch.Tensor(self.vec.vectors.shape[1])).unsqueeze(0),
                                    self.unk_init(torch.Tensor(self.vec.vectors.shape[1])).unsqueeze(0)],dim=0)

    def transform(self,X):
        if isinstance(X,tuple):
            X=list(X)
        if isinstance(X,list):
            return self.vec.get_vecs_by_tokens(X)
        else:
            return self.vec[X]
