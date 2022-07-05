import copy

from LAMDA_SSL.Transform.Vocab import Vocab

from sklearn.pipeline import Pipeline

from LAMDA_SSL.Transform.Tokenizer import Tokenizer
from LAMDA_SSL.Transform.Adjust_length import Adjust_length
from LAMDA_SSL.Transform.ToTensor import ToTensor
class TextMixin:
    def __init__(self,word_vocab=None,vectors=None,length=300,unk_token='<unk>',pad_token='<pad>',
                 min_freq=1,special_first=True,default_index=None):
        self.vectors=vectors
        self.word_vocab=word_vocab
        self.length=length
        self.unk_token=unk_token
        self.pad_token=pad_token
        self.min_freq=min_freq
        self.special_first=special_first
        self.specials=[self.unk_token,self.pad_token]
        self.default_index=default_index

    def init_default_transforms(self):

        if self.vectors is not None:
            self.vocab=Vocab(vectors=self.vectors.vec)

        else:
            if hasattr(self,'X'):
                text=copy.copy(self.X)
            elif hasattr(self,'labeled_X'):
                text=copy.copy(self.labeled_X)
            else:
                text=None

            self.vocab=Vocab(text=text,word_vocab=self.word_vocab,
                             min_freq=self.min_freq,
                             special_first=self.special_first,
                             specials=self.specials,
                             default_index=self.default_index)

            # self.transform=Pipeline([('Tokenizer',Tokenizer('basic_english')),
            #                   ('Adjust_length',Adjust_length(length=300)),
            #                   ('Word_vocab', Vocab(text=text,word_vocab=self.word_vocab,
            #                                        min_freq=self.min_freq,
            #                                        special_first=self.special_first,
            #                                        specials=self.specials,
            #                                        default_index=self.default_index,
            #                                        )),
            #                   ('ToTensor',ToTensor())
            #                   ])
        self.transform = Pipeline([
                                   ('Adjust_length', Adjust_length(length=self.length)),
                                   ('Vocab', self.vocab),
                                   ('ToTensor', ToTensor())
                                   ])
        self.pre_transform=Tokenizer('basic_english')
        self.valid_transform=copy.deepcopy(self.transform)
        self.test_transform=copy.deepcopy(self.transform)
        self.unlabeled_transform=copy.deepcopy(self.transform)
        self.target_transfrom=None
        self.transforms=None