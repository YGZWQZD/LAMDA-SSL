from torchtext.vocab import vocab
from collections import Counter,OrderedDict
from Semi_sklearn.Transform.Transformer import Transformer

class Vocab(Transformer):
    def __init__(self,text,min_freq=1,specials=["<unk>","<pad>"],special_first=True,word_vocab=None,default_index=None):
        super(Vocab, self).__init__()
        self.text=text
        self.specials=specials
        self.min_freq=min_freq
        self.special_first=special_first
        self.word_vocab=word_vocab
        if self.word_vocab is None:
            counter = Counter()
            for item in text:
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
        return self.word_vocab(X)

