import torch
from torch_geometric.nn import  GCNConv
import torch.nn.functional as F
class GCN(torch.nn.Module):
    def __init__(self,dim_in,num_classes,dim_hidden=16,normalize=False):
        # >> Parameter
        # >> - dim_in: The number of features.
        # >> - num_classes: The number of classes.
        # >> - normalize: Whether to add self-loops and compute symmetric normalization coefficients on the fly.
        super().__init__()
        self.conv1 = GCNConv(dim_in, dim_hidden, cached=True,
                             normalize=normalize)
        self.conv2 = GCNConv(dim_hidden, num_classes, cached=True,
                             normalize=normalize)

    def forward(self,data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x