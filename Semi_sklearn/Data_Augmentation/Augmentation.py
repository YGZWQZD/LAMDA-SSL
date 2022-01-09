from abc import abstractmethod,ABC
from sklearn.base import BaseEstimator, TransformerMixin
class Augmentation(BaseEstimator,TransformerMixin,ABC):
    def __init__(self):
        pass
    def fit(self,X=None,y=None,dataset=None):
        pass

    @abstractmethod
    def transform(self,X=None,y=None,dataset=None):
        if X is None and y is None and dataset is None:
            raise ValueError('No data to transform')
        raise NotImplementedError('Transform method of Augmentation class must be implemented.')

    @abstractmethod
    def fit_transform(self,X=None,y=None,dataset=None):
        if X is None and y is None and dataset is None:
            raise ValueError('No data to fit_transform')
        raise NotImplementedError('Fit_transform method of Augmentation class must be implemented.')
