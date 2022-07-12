from LAMDA_SSL.Base.Transformer import Transformer
import random
from LAMDA_SSL.Transform.Text.Tokenizer import Tokenizer

class RandomSwap(Transformer):
    def __init__(self,n=1,tokenizer=None):
        # >> Parameter:
        # >> - n: The number of times to swap words.
        # >> - tokenizer: The tokenizer used when the text is not untokenized.
        super(RandomSwap, self).__init__()
        self.n=n
        self.tokenizer=tokenizer if tokenizer is not None else Tokenizer('basic_english','en')

    def swap(self,X):
        random_idx_1 = random.randint(0, len(X) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(X) - 1)
            counter += 1
            if counter > 3:
                return X
        X[random_idx_1], X[random_idx_2] = X[random_idx_2], X[random_idx_1]
        return X

    def transform(self,X):
        tokenized=True
        if isinstance(X, str):
            X = self.tokenizer.fit_transform(X)
            tokenized = False
        for _ in range(self.n):
            X = self.swap(X)
        if tokenized is not True:
            X=' '.join(X)
        return X
