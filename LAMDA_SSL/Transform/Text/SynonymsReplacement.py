from LAMDA_SSL.Base.Transformer import Transformer
from LAMDA_SSL.Transform.Text.PadSequence import PadSequence
from LAMDA_SSL.Transform.Text.Truncate import Truncate
from synonyms import synonyms
import copy
import random
def synonym_replacement(tokens, n=10):
    new_tokens = copy.copy(tokens) # 复制分词结果，以免改变原始数据
    n=min(n,len(new_tokens))
    random_index=random.sample(range(len(new_tokens)), n)
    for i in range(n):
        rand_word_index = random_index[i]
        words = synonyms.nearby(tokens[rand_word_index])[0]
        if len(words)>=1:
            new_word = words[1]
            new_tokens[rand_word_index] = new_word
    return ''.join(new_tokens)

class SynonymsReplacement(Transformer):
    def __init__(self,n=10):
        super().__init__()
        self.n=n

    def transform(self,X):
        X=synonym_replacement(X,self.n)
        return X
