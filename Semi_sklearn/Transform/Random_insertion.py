from Semi_sklearn.Transform.Transformer import Transformer
from Semi_sklearn.Transform.Synonyms import Synonyms
import random
class Random_insertion(Transformer):
    def __init__(self,n=1):
        super(Random_insertion, self).__init__()
        self.n=n

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
        for _ in range(self.n):
            self.add_word(X)
        return X