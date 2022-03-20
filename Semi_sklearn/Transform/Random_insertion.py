from Semi_sklearn.Transform.Transformer import Transformer
from Semi_sklearn.Test.Synonyms import Synonyms
import random
from Semi_sklearn.Transform.Tokenizer import Tokenizer

class Random_insertion(Transformer):
    def __init__(self,n=1,tokenizer=None):
        super(Random_insertion, self).__init__()
        self.n=n
        self.tokenizer=tokenizer if tokenizer is not None else Tokenizer('basic_english','en')

    def add_word(self,X):

        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = X[random.randint(0, len(X) - 1)]
            synonyms = Synonyms().fit_transform(X=random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(X) - 1)
        X.insert(random_idx, random_synonym)

    def transform(self,X):
        tokenized=True
        if isinstance(X, str):
            X = self.tokenizer.fit_transform(X)
            tokenized = False
        for _ in range(self.n):
            self.add_word(X)
        if tokenized is not True:
            X=' '.join(X)
        return X