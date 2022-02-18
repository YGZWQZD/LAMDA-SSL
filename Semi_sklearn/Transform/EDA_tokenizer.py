from Semi_sklearn.Transform.Transformer import Transformer
import re
class EDA_tokenzier(Transformer):
    def __init__(self):
        super(EDA_tokenzier, self).__init__()

    def get_only_chars(self,line):
        clean_line = ""

        line = line.replace("’", "")
        line = line.replace("'", "")
        line = line.replace("-", " ")  # replace hyphens with spaces
        line = line.replace("\t", " ")
        line = line.replace("\n", " ")
        line = line.lower()

        for char in line:
            if char in 'qwertyuiopasdfghjklzxcvbnm ':
                clean_line += char
            else:
                clean_line += ' '

        clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
        if clean_line[0] == ' ':
            clean_line = clean_line[1:]
        return clean_line

    def transform(self,X):
        sentence = self.get_only_chars(X)
        words = sentence.split(' ')
        words = [word for word in words if word is not '']
        return words