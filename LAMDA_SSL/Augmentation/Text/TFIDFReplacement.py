import math
import numpy as np
from LAMDA_SSL.Base.Transformer import Transformer
from collections import Counter,defaultdict
from LAMDA_SSL.Transform.Text.Tokenizer import Tokenizer
class TFIDFReplacement(Transformer):
    def __init__(self,text,p=0.7,tokenizer=None,cache_len=100000):
        # >> Parameter:
        # >> - text: The text that needs to be augmented.
        # >> - p: Basic replacement probability.
        # >> - tokenizer: The tokenizer used when the text is not untokenized.
        # >> - cache_len: buffer size of Random numbers.
        super(TFIDFReplacement, self).__init__()
        self.text=text
        self.p=p
        self.tokenizer=Tokenizer('basic_english','en') if tokenizer is None else tokenizer
        self.cache_len=cache_len
        tot_counter=Counter()
        idf_counter=Counter()
        self.len_text=len(text)
        self.len_tot=0
        for line in text:
            if isinstance(line,str):
                line=self.tokenizer.fit_transform(line)
            cur_counter=Counter()
            tot_counter.update(line)
            cur_counter.update(line)
            for key,val in cur_counter.items():
                idf_counter.update([key])
                self.len_tot+=val

        self.idf={k: math.log((self.len_text+1)/(v+1)) for k,v in idf_counter.items()}
        self.tf={k:1.0*v/self.len_tot for k,v in tot_counter.items()}
        self.tf_idf_keys = []
        self.tf_idf_values = []
        self.tfidf={}
        for k, v, in idf_counter.items():
            self.tfidf[k]=self.idf[k]*self.tf[k]
            self.tf_idf_keys.append(k)
            self.tf_idf_values.append(self.tfidf[k])
        self.normalized_tf_idf = np.array(self.tf_idf_values)
        self.normalized_tf_idf = (self.normalized_tf_idf.max()
                                  - self.normalized_tf_idf)
        self.normalized_tf_idf = (self.normalized_tf_idf
                                  / self.normalized_tf_idf.sum())
        self.reset_token_list()
        self.reset_random_prob()

    def reset_random_prob(self):
        """Generate many random numbers at the same time and cache them."""
        self.random_prob_cache = np.random.random(size=(self.cache_len,))
        self.random_prob_ptr = self.cache_len - 1

    def reset_token_list(self):
        cache_len = len(self.tf_idf_keys)
        token_list_idx = np.random.choice(
            cache_len, (cache_len,), p=self.normalized_tf_idf)
        self.token_list = []
        for idx in token_list_idx:
            self.token_list += [self.tf_idf_keys[idx]]
        self.token_ptr = len(self.token_list) - 1

    def get_random_prob(self):
        """Get a random number."""
        value = self.random_prob_cache[self.random_prob_ptr]
        self.random_prob_ptr -= 1
        if self.random_prob_ptr == -1:
            self.reset_random_prob()
        return value

    def get_random_token(self):
        """Get a random token."""
        token = self.token_list[self.token_ptr]
        self.token_ptr -= 1
        if self.token_ptr == -1:
            self.reset_token_list()
        return token

    def get_replace_prob(self, X):
        cur_tf_idf = defaultdict(int)
        for word in X:
            cur_tf_idf[word] += 1. / len(X) * self.idf[word] if word in self.idf.keys() else 1. / len(X) *math.log(self.len_text)
        replace_prob = []
        for word in X:
            replace_prob += [cur_tf_idf[word]]
        replace_prob = np.array(replace_prob)
        replace_prob = np.max(replace_prob) - replace_prob
        replace_prob = replace_prob / replace_prob.sum() *self.p * len(X)
        for _ in range(len(replace_prob)):
            replace_prob[_]=min(1,replace_prob[_])
        return replace_prob

    def replace_tokens(self, word_list, replace_prob):
        """Replace tokens in a sentence."""
        for i in range(len(word_list)):
            if self.get_random_prob() < replace_prob[i]:
                word_list[i] = self.get_random_token()
        return word_list

    def transform(self,X):
        tokenized=True
        if isinstance(X, str):
            X = self.tokenizer.fit_transform(X)
            tokenized = False
        replace_prob = self.get_replace_prob(X)
        X = self.replace_tokens(
            X,
            replace_prob
        )
        if not tokenized:
            X=' '.join(X)
        return X





