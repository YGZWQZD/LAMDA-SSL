from Semi_sklearn.Transform.Transformer import Transformer
import nltk
class Synonyms(Transformer):
    def __init__(self,n=None):
        super(Synonyms, self).__init__()
        self.n=n
        nltk.download('wordnet')
    def transform(self,X):
        synonyms = set()
        for syn in nltk.corpus.wordnet.synsets(X):
            for l in syn.lemmas():
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym)
        if X in synonyms:
            synonyms.remove(X)
        if self.n is None:
            return list(synonyms)
        else:
            return list(synonyms)[:self.n]