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
    def __init__(self, input_dim = 28 ** 2, output_dim = 10,device='cpu'):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList([
            LinearWeightNorm(input_dim, 1000),
            LinearWeightNorm(1000, 500),
            LinearWeightNorm(500, 250),
            LinearWeightNorm(250, 250),
            LinearWeightNorm(250, 250)]
        )
        self.final = LinearWeightNorm(250, output_dim, weight_scale=1)
        self.device=device


    def forward(self, x):
        x = x.view(-1, self.input_dim)
        noise = torch.randn(x.size()) * 0.3 if self.training else torch.Tensor([0]).to(self.device)

        x = x + Variable(noise, requires_grad = False)
        x_f=x
        for i in range(len(self.layers)):
            m = self.layers[i]
            x_f = F.relu(m(x))
            noise = torch.randn(x_f.size()) * 0.5 if self.training else torch.Tensor([0]).to(self.device)
            x = (x_f + Variable(noise, requires_grad = False))

        self.feature=x_f

        return self.final(x)


class Generator(nn.Module):
    def __init__(self, input_dim = 28 ** 2,hidden_dim=500,z_dim=100,device='cpu'):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.device=device
        self.fc1 = nn.Linear(z_dim, hidden_dim, bias = False)
        self.bn1 = nn.BatchNorm1d(hidden_dim, affine = False, eps=1e-6, momentum = 0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.bn2 = nn.BatchNorm1d(hidden_dim, affine = False, eps=1e-6, momentum = 0.5)
        self.fc3 = LinearWeightNorm(hidden_dim, input_dim, weight_scale = 1)
        self.bn1_b = Parameter(torch.zeros(hidden_dim))
        self.bn2_b = Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)

    def forward(self, batch_size,z=None):
        z = Variable(torch.rand(batch_size, self.z_dim), requires_grad = False).to(self.device) if z is None else z
        x = F.softplus(self.bn1(self.fc1(z)) + self.bn1_b)
        x = F.softplus(self.bn2(self.fc2(x)) + self.bn2_b)
        x = F.softplus(self.fc3(x))
        return x

class ImprovedGAN(nn.Module):
    def __init__(self, G=None, D=None,input_dim = 28 ** 2, output_dim = 10,z_dim=100,device='cpu'):
        super(ImprovedGAN, self).__init__()
        self.G = G if G is not None else Generator(input_dim = input_dim,z_dim=z_dim,device=device)
        self.D = D if D is not None else Discriminator(input_dim,output_dim,device=device)

    def forward(self, x):
        return self.D(x)
