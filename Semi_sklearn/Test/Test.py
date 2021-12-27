import numpy
import sys
sys.path.append("../../")
print(sys.path)
from Semi_sklearn.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin

class Fixmatch(InductiveEstimator):
    def fit(self,X,y,unlabled_X):
        print("train")
    def predict(self,X):
        print("predict")
    
class FixmatchClassifier(Fixmatch,ClassifierMixin):
    def fit(self,X,y,unlabled_X):
        super().fit(X,y,unlabled_X)
    def predict(self, X):
        super().predict(X)

a=FixmatchClassifier()
a.fit(1,2,3)

a.predict(1)