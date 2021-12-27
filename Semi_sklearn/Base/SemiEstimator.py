from sklearn.base import BaseEstimator,ClassifierMixin
from abc import ABC,abstractmethod
class SemiEstimator(ABC,BaseEstimator):
    @abstractmethod
    def fit(self,X,y,unlabled_X,**params):
        raise NotImplementedError(
            "Fit method of SemiEstimator class must be implemented."
        )




# class Fixmatch(InductiveEstimator):
#     def fit(self,X,y,unlabled_X):
#         print("train")
#     def predict(self,X):
#         print("predict")
    
# class FixmatchClassifier(Fixmatch,ClassifierMixin):
#     def fit(self,X,y,unlabled_X):
#         super().fit(X,y,unlabled_X)
#     def predict(self, X):
#         super().predict(X)

# a=FixmatchClassifier()
# a.fit(1,2,3)

# a.predict(1)






