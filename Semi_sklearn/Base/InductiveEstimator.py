from .SemiEstimator import SemiEstimator
from abc import abstractmethod

class InductiveEstimator(SemiEstimator):
    _semi_type='Inductive'
    @abstractmethod
    def predict(self,X,**params):
        raise NotImplementedError(
            "Predict method must be implemented."
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