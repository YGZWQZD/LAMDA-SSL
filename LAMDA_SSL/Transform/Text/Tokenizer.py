from LAMDA_SSL.Base.Transformer import Transformer
from torchtext.data.utils import get_tokenizer

class Tokenizer(Transformer):
    def __init__(self,tokenizer='basic_english',language='en'):
        # >> Parameter:
        # >> - tokenizer: Function name for word segmentation, such as basic_english, spacy, moses, toktok, revtok, subword, etc.
        # >> - language: The language of the text.
        super(Tokenizer, self).__init__()
        self.tokenizer=tokenizer
        self.language=language
        self.transformer=get_tokenizer(self.tokenizer)

    def transform(self,X):
        X=self.transformer(X)
        return X