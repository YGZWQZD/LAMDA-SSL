import torch
import copy
class SDNE(torch.nn.Module):
    def __init__(self, dim_in, hidden_layers, device="cpu"):
        # >> Parameter:
        # >> - input_dim: The dimension of the input samples.
        # >> - hidden_layers: The dimension of the hidden layers.
        # >> - device: The device to train the model.
        super(SDNE, self).__init__()
        self.device = device
        dim_in_copy = copy.copy(dim_in)
        self.dim_in = dim_in_copy
        layers = []
        for layer_dim in hidden_layers:
            layers.append(torch.nn.Linear(dim_in, layer_dim))
            layers.append(torch.nn.ReLU())
            dim_in = layer_dim
        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for layer_dim in reversed(hidden_layers[:-1]):
            layers.append(torch.nn.Linear(dim_in , layer_dim))
            layers.append(torch.nn.ReLU())
            dim_in  = layer_dim

        layers.append(torch.nn.Linear(dim_in , dim_in_copy))
        layers.append(torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(*layers)

    def forward(self,X=None):
        Y = self.encoder(X)
        X_hat = self.decoder(Y)
        return Y,X_hat
