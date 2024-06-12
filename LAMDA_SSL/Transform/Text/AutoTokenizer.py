from LAMDA_SSL.Base.Transformer import Transformer
from LAMDA_SSL.Transform.Text.PadSequence import PadSequence
from LAMDA_SSL.Transform.Text.Truncate import Truncate
from transformers import AutoTokenizer as AT

class AutoTokenizer(Transformer):
    def __init__(self,model_name='hfl/chinese-roberta-wwm-ext',padding='max_length',
                 truncation=True,max_length=256,return_tensors='pt',local_files_only=False):
        super().__init__()
        self.local_files_only=local_files_only
        self.tokenizer=AT.from_pretrained(model_name,local_files_only=self.local_files_only)
        self.padding=padding
        self.truncation=truncation
        self.max_length=max_length
        self.return_tensors=return_tensors

    def transform(self,X):
        X=self.tokenizer(X, padding=self.padding, truncation=self.truncation, 
            max_length=self.max_length, return_tensors=self.return_tensors)
        return X
