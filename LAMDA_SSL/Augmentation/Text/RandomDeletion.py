from LAMDA_SSL.Base.Transformer import Transformer
import random
from LAMDA_SSL.Transform.Text.Tokenizer import Tokenizer

class RandomDeletion(Transformer):
    def __init__(self,p=0.5,tokenizer=None):
        # >> Parameter:
        # >> - p: The proportion of random deletions.
        # >> - tokenizer: The tokenizer used when the text is not untokenized.
        super(RandomDeletion, self).__init__()
        self.p=p
        self.tokenizer=tokenizer if tokenizer is not None else Tokenizer('basic_english','en')

    def transform(self,X):
        tokenized=True
        if isinstance(X, str):
            X = self.tokenizer.fit_transform(X)
            tokenized = False
        # obviously, if there's only one word, don't delete it
        if len(X) == 1:
            return X

        # randomly delete words with probability p
        new_words = []
        for word in X:
            r = random.uniform(0, 1)
            if r > self.p:
                new_words.append(word)

        # if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(X) - 1)
            new_words=[X[rand_int]]
        if tokenized is not True:
            new_words=' '.join(new_words)
        return new_words
