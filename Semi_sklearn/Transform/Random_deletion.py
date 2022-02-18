from Semi_sklearn.Transform.Transformer import Transformer
import random
class Random_deletion(Transformer):
    def __init__(self,p):
        super(Random_deletion, self).__init__()
        self.p=p
    def transform(self,X):

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
            return [X[rand_int]]

        return new_words
