from abc import abstractmethod,ABC
from sklearn.base import BaseEstimator, TransformerMixin

class Transformer(BaseEstimator,TransformerMixin,ABC):
    def __init__(self):
        pass

    def fit(self,X,y=None,**fit_params):
        # >> fit(X,y=None): Obtain the processing function through existing data.
        # >> - X: Samples for learning the function of transformation.
        # >> - y: Labels for learning the function of transformation.
        return self

    def __call__(self, X,y=None,**fit_params):
        # >> __call__(X,y=None): It is the same as fit_transform.
        # >> - X: Samples for learning and transformation.
        # >> - y: labels for learning.
        return self.fit_transform(X,y,fit_params=fit_params)

    @abstractmethod
    def transform(self,X):
        # >> transform(X): Process the new data.
        # >> - X: Data to be converted.
        raise NotImplementedError('Transform method of Augmentation class must be implemented.')

    def fit_transform(self,X,y=None,**fit_params):
        # >> fit_transform(X,y=None): Firstly perform fit() on the existing samples X and labels y, and then directly transform y.
        # >> - X: Samples for learning and transformation.
        # >> - y: Labels fo learning
        return self.fit(X=X,y=y,fit_params=fit_params).transform(X)
