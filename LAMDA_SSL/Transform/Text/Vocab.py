from torchtext.vocab import vocab
from collections import Counter,OrderedDict
from LAMDA_SSL.Base.Transformer import Transformer
from LAMDA_SSL.Transform.Text.Tokenizer import Tokenizer

class Vocab(Transformer):
    def __init__(self,word_vocab=None,vectors=None,text=None,min_freq=1,specials=["<unk>","<pad>"],special_first=True,default_index=None,tokenizer=None):
        # >> Parameter:
        # >> - word_vocab: A map that converts words to indexes.
        # >> - vectors: Word vectors.
        # >> - text: When word_vocab is None, use text to create a mapping table.
        # >> - min_freq: The minimum frequency required for a word to be used as a token in the word_vocab. It is valid when word_vocab is None and a mapping table needs to be constructed.
        # >> - specials: List of special characters.
        # >> - special_first: Whether to put special characters at the top of the mapping table.
        # >> - default_index: The default value that should be used when converting a word to an index if it cannot be converted.
        # >> - tokenizer: The word segmentation method used.
        super(Vocab, self).__init__()
        self.text=text
        self.specials=specials
        self.min_freq=min_freq
        self.special_first=special_first
        self.word_vocab=word_vocab
        self.vectors=vectors
        self.tokenizer=Tokenizer('basic_english','en') if tokenizer is None else tokenizer
        if self.vectors is not None:
            self.word_vocab=self.vectors.stoi
            self.default_index = self.word_vocab["<unk>"] if default_index is None else default_index
        elif self.word_vocab is None:
            counter = Counter()
            for item in text:
                if isinstance(item ,str):
                    item=self.tokenizer.fit_transform(item)
                counter.update(item)
            if specials is not None:
                for tok in specials:
                    del counter[tok]
            sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
            sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
            ordered_dict = OrderedDict(sorted_by_freq_tuples)
            if specials is not None:
                if special_first:
                    specials = specials[::-1]
                for symbol in specials:
                    ordered_dict.update({symbol: min_freq})
                    ordered_dict.move_to_end(symbol, last=not special_first)
            self.word_vocab = vocab(ordered_dict, min_freq=min_freq)
            self.default_index = self.word_vocab["<unk>"] if default_index is None else default_index
            self.word_vocab.set_default_index(self.default_index)



    def transform(self,X):
        if self.vectors is not None:
            return [self.word_vocab[item] if item in self.word_vocab.keys() else self.default_index for item in X]
        return [self.word_vocab[item] for item in X]

