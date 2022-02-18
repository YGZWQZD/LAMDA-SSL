from Semi_sklearn.Transform.Transformer import Transformer
from Semi_sklearn.Transform.EDA_tokenizer import EDA_tokenzier
import random
from Semi_sklearn.Transform.Random_deletion import Random_deletion
from Semi_sklearn.Transform.Synonym_replacement import Synonym_replacement
from Semi_sklearn.Transform.Random_insertion import Random_insertion
from Semi_sklearn.Transform.Random_swap import Random_swap

class EDA(Transformer):
    def __init__(self,alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=1,tokenizer=None):
        super().__init__()
        self.alpha_sr=alpha_sr
        self.alpha_ri=alpha_ri
        self.alpha_rs=alpha_rs
        self.p_rd=p_rd
        self.num_aug=num_aug
        self.num_new_per_technique = int(num_aug/4)+1
        self.tokenizer=tokenizer if tokenizer is not None else EDA_tokenzier()
    def transform(self,X):
        tokenized=True
        if isinstance(X, str):
            X = self.tokenizer.fit_transform(X)
            tokenized = False

        augmented_sentences = []
        # sr
        num_words = len(X)
        if (self.alpha_sr > 0):
            n_sr = max(1, int(self.alpha_sr *num_words))
            for _ in range(self.num_new_per_technique):
                a_words = Synonym_replacement(n_sr).fit_transform(X)
                augmented_sentences.append(a_words)

        # ri
        if (self.alpha_ri > 0):
            n_ri = max(1, int(self.alpha_ri * num_words))
            for _ in range(self.num_new_per_technique):
                a_words = Random_insertion(n_ri).fit_transform(X)
                augmented_sentences.append(a_words)

        # rs
        if (self.alpha_rs > 0):
            n_rs = max(1, int(self.alpha_rs * num_words))
            for _ in range(self.num_new_per_technique):
                a_words = Random_swap(n_rs).fit_transform(X)
                augmented_sentences.append(a_words)

        # rd
        if (self.p_rd > 0):
            for _ in range(self.num_new_per_technique):
                a_words = Random_deletion(self.p_rd).fit_transform(X)
                augmented_sentences.append(a_words)

        random.shuffle(augmented_sentences)
        if tokenized is not True:
            augmented_sentences=[' '.join(item) for item in augmented_sentences]
        # append the original sentence
        result=augmented_sentences[:self.num_aug]
        if self.num_aug==1:
            result=result[0]

        return result