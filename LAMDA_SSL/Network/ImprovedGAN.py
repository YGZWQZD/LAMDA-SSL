import numbers

import torch
from torch.nn.parameter import Parameter
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class LinearWeightNorm(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_scale=None, weight_init_stdv=0.1):
        super(LinearWeightNorm, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.randn(out_features, in_features) * weight_init_stdv)
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if weight_scale is not None:
            assert type(weight_scale) == int
            self.weight_scale = Parameter(torch.ones(out_features, 1) * weight_scale)
        else:
            self.weight_scale = 1
    def forward(self, x):
        W = self.weight * self.weight_scale / torch.sqrt(torch.sum(self.weight ** 2, dim = 1, keepdim = True))
        return F.linear(x, W, self.bias)

class Discriminator(nn.Module):
    def __init__(self, dim_in = 28 ** 2,hidden_dim=[1000,500,250,250,250],
                 noise_level=[0.3,0.5,0.5,0.5,0.5,0.5],activations=[nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU()],
                 dim_out = 10,device='cpu'):
        super(Discriminator, self).__init__()
        self.dim_in = dim_in
        self.num_hidden=len(hidden_dim)
        self.layers = torch.nn.ModuleList()
        self.noise_level=noise_level
        for _ in range(self.num_hidden):
            if _==0:
                input_dim=dim_in
            else:
                input_dim=hidden_dim[_-1]
            out_dim=hidden_dim[_]
            self.layers.append(LinearWeightNorm(input_dim, out_dim))
        self.final = LinearWeightNorm(hidden_dim[self.num_hidden-1], dim_out, weight_scale=1)
        self.activations=activations
        self.device=device


    def forward(self, x):
        x = x.view(-1, self.dim_in)
        noise = torch.randn(x.size()).to(self.device) * self.noise_level[0] if self.training else torch.Tensor([0]).to(self.device)

        x = x + Variable(noise, requires_grad = False)
        x_f=x
        for i in range(len(self.layers)):
            m = self.layers[i]
            x_f = self.activations[i](m(x))
            noise = torch.randn(x_f.size()).to(self.device) * self.noise_level[i+1] if self.training else torch.Tensor([0]).to(self.device)
            x = (x_f + Variable(noise, requires_grad = False))

        self.feature=x_f
        if len(self.activations)==self.num_hidden+1:
            x = self.activations[self.num_hidden](self.final(x))
        else:
            x=self.final(x)
        return x


class Generator(nn.Module):
    def __init__(self, dim_in = 28 ** 2,hidden_dim=[500,500],activations=[nn.Softplus(),nn.Softplus(),nn.Softplus()],dim_z=100,device='cpu'):
        super(Generator, self).__init__()
        self.dim_z = dim_z
        self.device=device
        self.hidden_dim=hidden_dim
        self.layers = torch.nn.ModuleList()
        self.bn_layers=torch.nn.ModuleList()
        self.bn_b = torch.nn.ParameterList()
        self.num_hidden=len(hidden_dim)
        self.activations=activations
        for _ in range(self.num_hidden):
            if _==0:
                input_dim=dim_z
            else:
                input_dim=hidden_dim[_-1]
            output_dim=hidden_dim[_]
            fc=nn.Linear(input_dim, output_dim, bias=False)
            nn.init.xavier_uniform(fc.weight)
            self.layers.append(fc)
            self.bn_layers.append(nn.BatchNorm1d(output_dim, affine = False, eps=1e-6, momentum = 0.5))
            self.bn_b.append(Parameter(torch.zeros(output_dim)))
        self.fc = LinearWeightNorm(hidden_dim[self.num_hidden-1], dim_in, weight_scale = 1)


    def forward(self, batch_size=10,z=None):
        z = Variable(torch.rand(batch_size, self.dim_z), requires_grad = False,volatile = not self.training).to(self.device) if z is None else z
        for _ in range(self.num_hidden):
            z = self.activations[_](self.bn_layers[_](self.layers[_](z)) + self.bn_b[_])
        if len(self.activations)==self.num_hidden+1:
            z = self.activations[self.num_hidden](self.fc(z))
        else:
            z=self.fc(z)
        return z

class ImprovedGAN(nn.Module):
    def __init__(self, G=None, D=None,dim_in = 28 ** 2,
                 hidden_G=[1000,500,250,250,250],
                 hidden_D=[1000,500,250,250,250],
                 noise_level=[0.3, 0.5, 0.5, 0.5, 0.5, 0.5],
                 activations_G=[nn.Softplus(), nn.Softplus(), nn.Softplus(),nn.Softplus(), nn.Softplus(), nn.Softplus()],
                 activations_D=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
                 dim_out = 10,dim_z=100,device='cpu'):
        # >> Parameter
        # >> - G: The neural network of generator.
        # >> - D: The neural network of discriminator
        # >> - dim_in: The dimension of the inputted samples.
        # >> - hidden_G: The dimension of the generator's hidden layers.
        # >> - hidden_D: The dimension of the discriminator's hidden layers.
        # >> - activations_G: The activation functions for each layer of the generator.
        # >> - activations_D: The activation functions for each layer of the discriminator.
        # >> - output_dim: The dimension of outputs.
        # >> - z_dim: The dimension of the hidden variable used to generate data.
        # >> - device: The device to train the model.
        super(ImprovedGAN, self).__init__()
        if isinstance(dim_in,numbers.Number):
            input_dim = dim_in
        else:
            input_dim=1
            for item in dim_in:
                input_dim=input_dim*item
        self.G = G if G is not None else Generator(dim_in = input_dim,dim_z=dim_z,
                                                   hidden_dim=hidden_G,activations=activations_G,
                                                   device=device)
        self.D = D if D is not None else Discriminator(dim_in=input_dim,noise_level=noise_level,
                                                       activations=activations_D,dim_out=dim_out,
                                                       hidden_dim=hidden_D,device=device)

    def forward(self, x):
        return self.D(x)
