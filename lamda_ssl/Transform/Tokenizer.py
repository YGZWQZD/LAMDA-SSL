from lamda_ssl.Transform.Transformer import Transformer
from torchtext.data.utils import get_tokenizer

class Tokenizer(Transformer):
    '''
    Generate tokenizer function for a string sentence.

    Args:
        tokenizer: the name of tokenizer function. If None, it returns split()
            function, which splits the string sentence by space.
            If basic_english, it returns _basic_english_normalize() function,
            which normalize the string first and split by space. If a callable
            function, it will return the function. If a tokenizer library
            (e.g. spacy, moses, toktok, revtok, subword), it returns the
            corresponding library.
        language: Default en
    '''
    def __init__(self,tokenizer,language='en'):
        super(Tokenizer, self).__init__()
        self.tokenizer=tokenizer
        self.language=language
        self.transformer=get_tokenizer(self.tokenizer)

    def transform(self,X):
        X=self.transformer(X)
        return X