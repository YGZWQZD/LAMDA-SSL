from torchtext import vocab
from LAMDA_SSL.Base.Transformer import Transformer
import torch
class Vectors(Transformer):
    def __init__(self,name='840B', cache=None, url=None, unk_init=None,pad_init=None, max_vectors=None,lower_case_backup=True,
                 pad_token='<pad>',unk_token='<unk>'):
        # >> Parameter:
        # >> - name: The name of the word vector.
        # >> - cache: Directory for cached vectors。
        # >> - url: The download address of the word vector.
        # >> - unk_init: By default, initialize out-of-vocabulary word vectors to zero vectors; can be any function that takes in a Tensor and returns a Tensor of the same size.
        # >> - pad_init: By default, initialize out-of-vocabulary word vectors to zero vectors; can be any function that takes in a Tensor and returns a Tensor of the same size.
        # >> - max_vectors: The maximum number of word vectors.
        # >> - lower_case_backup: Whether to convert all to lowercase when looking up words.
        # >> - pad_token: The default padding token.
        # >> - unk_token: The default token represents unknown words.
        super(Vectors, self).__init__()
        self.vec=vocab.Vectors(name,cache,url,unk_init,max_vectors)
        self.unk_init = torch.Tensor.zero_ if unk_init is None else unk_init
        self.pad_init = torch.Tensor.zero_ if pad_init is None else pad_init
        self.lower_case_backup=lower_case_backup
        self.vec.stoi[pad_token]=self.vec.vectors.shape[0]
        self.vec.stoi[unk_token] = self.vec.vectors.shape[0]+1
        self.vec.vectors=torch.cat([self.vec.vectors,self.pad_init(torch.Tensor(self.vec.vectors.shape[1])),
                                    self.unk_init(torch.Tensor(self.vec.vectors.shape[1]))],dim=0)



    def transform(self,X):
        if isinstance(X,tuple):
            X=list(X)
        if isinstance(X,list):
            if isinstance(X[0],str):
                return self.vec.get_vecs_by_tokens(X,lower_case_backup=self.lower_case_backup)
            else:
                indices = [self.vec.vectors[idx] for idx in X]
                return torch.stack(indices)
        else:
            if isinstance(X[0],str):
                return self.vec[X]
            else:
                return self.vec.vectors[X]

