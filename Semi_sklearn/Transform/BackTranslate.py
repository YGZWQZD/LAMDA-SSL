from Semi_sklearn.Transform.Transformer import Transformer
from googletrans import Translator
class Translate(Transformer):
    def __init__(self,sorce='zh-CN',target='zh-CN'):
        super().__init__()
        self.source=sorce
        self.target=target
        self.translator=Translator()
    def transform(self,X):
        X = self.translator.translate(X, dest=self.target, src=self.source).text
        X = self.translator.translate(X, dest=self.source, src=self.target).text
        return X
