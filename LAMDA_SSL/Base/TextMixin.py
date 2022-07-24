import copy
from LAMDA_SSL.Transform.Text.Vocab import Vocab
from sklearn.pipeline import Pipeline
from LAMDA_SSL.Transform.Text.Tokenizer import Tokenizer
from LAMDA_SSL.Transform.Text.AdjustLength import AdjustLength
from LAMDA_SSL.Transform.ToTensor import ToTensor
class TextMixin:
    def __init__(self,word_vocab=None,vectors=None,length=300,unk_token='<unk>',pad_token='<pad>',
                 min_freq=1,special_first=True,default_index=None):
        # >> parameter:
        # >> - word_vocab: A map that converts words to indexes.
        # >> - vectors: Word vectors.
        # >> - length: Length of each sentence.
        # >> - unk_token: The token used to represent unknown words.
        # >> - pad_token: The token used to represent padding.
        # >> - min_freq: The minimum frequency required for a word to be used as a token in the word_vocab. It is valid when word_vocab is None and a mapping table needs to be constructed.
        # >> - special_first: Whether to put special characters at the top of the mapping table.
        # >> - default_index: The default value that should be used when converting a word to an index if it cannot be converted.
        # >> init_default_transform: Initialize the data transformation method.
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

        self.pre_transform = Tokenizer('basic_english')
        self.transform = Pipeline([('Adjust_length', AdjustLength(length=self.length)),
                                   ('Vocab', self.vocab),
                                   ('ToTensor', ToTensor())
                                   ])
        self.valid_transform=Pipeline([('Adjust_length', AdjustLength(length=self.length)),
                                   ('Vocab', self.vocab),
                                   ('ToTensor', ToTensor())
                                   ])
        self.test_transform=Pipeline([('Adjust_length', AdjustLength(length=self.length)),
                                   ('Vocab', self.vocab),
                                   ('ToTensor', ToTensor())
                                   ])
        self.unlabeled_transform=Pipeline([('Adjust_length', AdjustLength(length=self.length)),
                                   ('Vocab', self.vocab),
                                   ('ToTensor', ToTensor())
                                   ])
        self.target_transfrom=None
        self.transforms=None