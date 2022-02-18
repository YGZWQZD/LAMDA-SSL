from Semi_sklearn.Transform.Transformer import Transformer
from Semi_sklearn.Transform.Synonyms import Synonyms
import random
class Synonym_replacement(Transformer):

    def __init__(self,n=1):
        super(Synonym_replacement, self).__init__()
        self.n=n
        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
                  'ours', 'ourselves', 'you', 'your', 'yours',
                  'yourself', 'yourselves', 'he', 'him', 'his',
                  'himself', 'she', 'her', 'hers', 'herself',
                  'it', 'its', 'itself', 'they', 'them', 'their',
                  'theirs', 'themselves', 'what', 'which', 'who',
                  'whom', 'this', 'that', 'these', 'those', 'am',
                  'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'having', 'do', 'does', 'did',
                  'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                  'because', 'as', 'until', 'while', 'of', 'at',
                  'by', 'for', 'with', 'about', 'against', 'between',
                  'into', 'through', 'during', 'before', 'after',
                  'above', 'below', 'to', 'from', 'up', 'down', 'in',
                  'out', 'on', 'off', 'over', 'under', 'again',
                  'further', 'then', 'once', 'here', 'there', 'when',
                  'where', 'why', 'how', 'all', 'any', 'both', 'each',
                  'few', 'more', 'most', 'other', 'some', 'such', 'no',
                  'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                  'very', 's', 't', 'can', 'will', 'just', 'don',
                  'should', 'now', '']

    def transform(self,X):
        random_word_list = list(set([word for word in X if word not in self.stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = Synonyms().fit_transform(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                X = [synonym if word == random_word else word for word in X]
                # print("replaced", random_word, "with", synonym)
                num_replaced += 1
            if num_replaced >= self.n:  # only replace up to n words
                break

        # this is stupid but we need it, trust me
        sentence = ' '.join(X)
        result = sentence.split(' ')

        return result