from LAMDA_SSL.Base.Transformer import Transformer
from sklearn.utils import check_random_state
import numpy as np
class DropNodes(Transformer):
    def __init__(self,num_drop,shuffle=True,random_state=None):
        # >> Parameter:
        # >> - num_drop: The number of nodes to be dropped.
        # >> - shuffle: Whether to shuffle the data.
        # >> - random_state: The random seed.
        super().__init__()
        self.num_drop=num_drop
        self.shuffle=shuffle
        self.random_state=random_state

    def transform(self,X):
        num_nodes=X.num_nodes
        num_edges=X.num_edges
        if self.shuffle:
            rng=check_random_state(self.random_state)
            permutation = rng.permutation(num_nodes)
        else:
            permutation = np.arange(num_nodes)
        ind_drop_node = permutation[:self.num_drop].tolist()
        ind_save_node = permutation[self.num_drop : num_nodes].tolist()

        ind_drop_edge = []
        ind_save_edge = []

        edge_index=X.edge_index

        for _ in np.arange(num_edges):
            s,t=edge_index[0][_],edge_index[1][_]
            if s in ind_drop_node or t in ind_drop_node:
                ind_drop_edge.append(_)
            else:
                ind_save_edge.append(_)

        if hasattr(X,'x')and X.x is not None:
            X.x=X.x[ind_save_node]
        if hasattr(X, 'train_mask')and X.train_mask is not None:
            X.train_mask=X.train_mask[ind_save_node]
        if hasattr(X, 'labeled_mask')and X.labeled_mask is not None:
            X.labeled_mask = X.labeled_mask[ind_save_node]
        if hasattr(X, 'unlabeled_mask')and X.unlabeled_mask is not None:
            X.unlabeled_mask=X.unlabeled_mask[ind_save_node]
        if hasattr(X, 'val_mask')and X.valid_mask is not None:
            X.valid_mask=X.val_mask[ind_save_node]
        if hasattr(X, 'test_mask')and X.test_mask is not None:
            X.test_mask=X.test_mask[ind_save_node]
        if hasattr(X,'edge_index')and X.edge_index is not None:
            X.edge_index=X.edge_index[:,ind_save_edge]
        if hasattr(X, 'edge_attr') and X.edge_attr is not None:
            X.edge_attr = X.edge_attr[:, ind_save_edge]
        if hasattr(X,'edge_weight') and X.edge_weight is not None:
            X.edge_weight=X.edge_weight[ind_save_edge]

        X.num_nodes=len(ind_save_node)
        X.num_edges=len(ind_save_edge)

        return X