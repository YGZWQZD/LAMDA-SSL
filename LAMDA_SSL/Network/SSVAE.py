import numbers
import torch
import torch.nn as nn
import torch.distributions as DT

class SSVAE(nn.Module):
    """
    Data model (SSL paper eq 2):
        p(y) = Cat(y|pi)
        p(z) = Normal(z|0,1)
        p(x|y,z) = f(x; z,y,theta)
    Recognition model / approximate posterior q_phi (SSL paper eq 4):
        q(y|x) = Cat(y|pi_phi(x))
        q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x)))
    """
    def __init__(self, dim_in,num_classes,dim_z,dim_hidden_de=[500,500],
                 dim_hidden_en_y=[500,500],dim_hidden_en_z=[500,500],
                 activations_de=[nn.Softplus(),nn.Softplus()],
                 activations_en_y=[nn.Softplus(),nn.Softplus()],
                 activations_en_z=[nn.Softplus(),nn.Softplus()],
                 device='cpu'):
        # >> Parameter:
        # >> - dim_in: The dimension of the input sample.
        # >> - num_classes: The number of classes.
        # >> - dim_z: The dimension of the hidden variable z.
        # >> - dim_hidden_de: The hidden layer dimension of the decoder.
        # >> - dim_hidden_en_y: The hidden layer dimension of the encoder for y.
        # >> - dim_hidden_en_z: The hidden layer dimension of the encoder for z.
        # >> - activations_de: The activation functions of the decoder.
        # >> - activations_en_y: The activation functions of the encoder for y.
        # >> - activations_en_z: The activation functions of the encoder for z.
        # >> - device: The device to train the model.
        super().__init__()
        if isinstance(dim_in,numbers.Number):
            input_dim = dim_in
        else:
            input_dim=1
            for item in dim_in:
                input_dim=input_dim*item

        self.p_y = DT.OneHotCategorical(probs=1 / num_classes * torch.ones(1,num_classes, device=device))
        self.p_z = DT.Normal(torch.tensor(0., device=device), torch.tensor(1., device=device))

        # parametrized data likelihood p(x|y,z)
        num_hidden=len(dim_hidden_de)

        self.decoder = nn.Sequential()
        for _ in range(num_hidden):
            if _==0:
                in_dim=dim_z + num_classes
            else:
                in_dim=dim_hidden_de[_-1]
            out_dim=dim_hidden_de[_]
            name = "Linear_" + str(_)
            self.decoder.add_module(name=name,module=nn.Linear(in_dim,out_dim))
            name = "Activation_" + str(_)
            self.decoder.add_module(name=name,module=activations_de[_])
        name = "Linear_" + str(num_hidden)
        in_dim = dim_hidden_de[num_hidden- 1]
        out_dim=input_dim
        self.decoder.add_module(name=name, module=nn.Linear(in_dim,out_dim))
        name = "BatchNorm"
        self.decoder.add_module(name=name, module=nn.BatchNorm1d(out_dim, affine=False))
        self.encoder_y = nn.Sequential()
        num_hidden = len(dim_hidden_en_y)
        for _ in range(num_hidden):
            if _==0:
                in_dim=input_dim
            else:
                in_dim=dim_hidden_en_y[_-1]
            out_dim=dim_hidden_en_y[_]
            name = "Linear_" + str(_)
            self.encoder_y.add_module(name=name,module=nn.Linear(in_dim,out_dim))
            name = "Activation_" + str(_)
            self.encoder_y.add_module(name=name,module=activations_en_y[_])
        name = "Linear_" + str(num_hidden)
        in_dim = dim_hidden_en_y[num_hidden- 1]
        out_dim=num_classes
        self.encoder_y.add_module(name=name, module=nn.Linear(in_dim,out_dim))

        # parametrized q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x)))
        self.encoder_z = nn.Sequential()
        num_hidden = len(dim_hidden_en_z)
        for _ in range(num_hidden):
            if _==0:
                in_dim=input_dim + num_classes
            else:
                in_dim=dim_hidden_en_z[_-1]
            out_dim=dim_hidden_en_z[_]
            name = "Linear_" + str(_)
            self.encoder_z.add_module(name=name,module=nn.Linear(in_dim,out_dim))
            name = "Activation_" + str(_)
            self.encoder_z.add_module(name=name,module=activations_en_z[_])
        name = "Linear_" + str(num_hidden)
        in_dim = dim_hidden_en_z[num_hidden- 1]
        out_dim=2 * dim_z
        self.encoder_z.add_module(name=name, module=nn.Linear(in_dim,out_dim))

        for p in self.parameters():
            p.data.normal_(0, 0.001)
            if p.ndimension() == 1: p.data.fill_(0.)

    # q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x))) -- SSL paper eq 4
    def encode_z(self, x, y):
        xy = torch.cat([x, y], dim=1)
        mu, logsigma = self.encoder_z(xy).chunk(2, dim=-1)
        return DT.Normal(mu, logsigma.exp())

    # q(y|x) = Categorical(y|pi_phi(x)) -- SSL paper eq 4
    def encode_y(self, x):
        return DT.OneHotCategorical(logits=self.encoder_y(x))

    # p(x|y,z) = Bernoulli
    def decode(self, y, z):
        yz = torch.cat([y, z], dim=1)
        reconstruction = DT.Bernoulli(logits=self.decoder(yz))
        return reconstruction

    # classification model q(y|x) using the trained q distribution
    def forward(self, x):
        y_probs = self.encode_y(x).probs
        return y_probs