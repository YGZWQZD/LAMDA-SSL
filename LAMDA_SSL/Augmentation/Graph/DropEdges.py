from LAMDA_SSL.Base.Transformer import Transformer
from sklearn.utils import check_random_state
import numpy as np


class DropEdges(Transformer):
    def __init__(self, num_drop, shuffle=True, random_state=None):
        # >> Parameter:
        # >> - num_drop: The number of edges to be dropped.
        # >> - shuffle: Whether to shuffle the data.
        # >> - random_state: The random seed.
        super().__init__()
        self.num_drop = num_drop
        self.shuffle = shuffle
        self.random_state = random_state

    def transform(self, X):
        num_edges = X.num_edges
        if self.shuffle:
            rng = check_random_state(self.random_state)
            permutation = rng.permutation(num_edges)
        else:
            permutation = np.arange(num_edges)
        ind_drop_edge = permutation[:self.num_drop].tolist()
        ind_save_edge= permutation[self.num_drop: num_edges].tolist()

        if hasattr(X, 'edge_index')and X.edge_index is not None:
            X.edge_index = X.edge_index[:,ind_save_edge]
        if hasattr(X, 'edge_weight')and X.edge_weight is not None:
            X.edge_weight = X.edge_weight[ind_save_edge]
        if hasattr(X, 'edge_attr')and X.edge_attr is not None:
            X.edge_attr = X.edge_attr[ind_save_edge]
        X.num_edges = len(ind_save_edge)

        return X