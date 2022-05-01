import numbers
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.distributions as DT
import torch.nn.functional as F
from torch.nn import init
# class GaussianSample(nn.Module):
#     """
#     Layer that represents a sample from a
#     Gaussian distribution.
#     """
#     def __init__(self, in_features, out_features):
#         super(GaussianSample, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#
#         self.mu = nn.Linear(in_features, out_features)
#         self.log_var = nn.Linear(in_features, out_features)
#
#     def reparametrize(self, mu, log_var):
#         epsilon = Variable(torch.randn(mu.size()), requires_grad=False)
#
#         if mu.is_cuda:
#             epsilon = epsilon.cuda()
#
#         # log_std = 0.5 * log_var
#         # std = exp(log_std)
#         std = log_var.mul(0.5).exp_()
#
#         # z = std * epsilon + mu
#         z = mu.addcmul(std, epsilon)
#
#         return z
#
#     def forward(self, x):
#         mu = self.mu(x)
#         log_var = F.softplus(self.log_var(x))
#
#         return self.reparametrize(mu, log_var), mu, log_var

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
    def __init__(self, dim_in,num_class,dim_z,dim_hidden_de=[500,500],
                 dim_hidden_en_y=[500,500],dim_hidden_en_z=[500,500],
                 activations_de=[nn.Softplus(),nn.Softplus()],
                 activations_en_y=[nn.Softplus(),nn.Softplus()],
                 activations_en_z=[nn.Softplus(),nn.Softplus()],
                 device='cpu'):
        super().__init__()
        if isinstance(dim_in,numbers.Number):
            input_dim = dim_in
        else:
            input_dim=1
            for item in dim_in:
                input_dim=input_dim*item

        # --------------------
        # p model -- SSL paper generative semi supervised model M2
        # --------------------

        self.p_y = DT.OneHotCategorical(probs=1 / num_class * torch.ones(1,num_class, device=device))
        self.p_z = DT.Normal(torch.tensor(0., device=device), torch.tensor(1., device=device))

        # parametrized data likelihood p(x|y,z)
        num_hidden=len(dim_hidden_de)

        self.decoder = nn.Sequential()
        for _ in range(num_hidden):
            if _==0:
                in_dim=dim_z + num_class
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
        # nn.Linear(dim_z + num_class, dim_hidden),
        # nn.Softplus(),
        # nn.Linear(dim_hidden, dim_hidden),
        # nn.Softplus(),
        # # nn.Linear(dim_hidden, dim_hidden),
        # # nn.Softplus(),
        # nn.Linear(dim_hidden, input_dim * 2),
        # torch.nn.BatchNorm1d(input_dim * 2, affine=False)

        # --------------------
        # q model -- SSL paper eq 4
        # --------------------

        # parametrized q(y|x) = Cat(y|pi_phi(x)) -- outputs parametrization of categorical distribution
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
        out_dim=num_class
        self.encoder_y.add_module(name=name, module=nn.Linear(in_dim,out_dim))
        # nn.Linear(input_dim, dim_hidden),
        # nn.Softplus(),
        # nn.Linear(dim_hidden, dim_hidden),
        # # nn.ReLU(),
        # # nn.Linear(dim_hidden, dim_hidden),
        # nn.Softplus(),
        # nn.Linear(dim_hidden, num_class)

        # parametrized q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x))) -- output parametrizations for mean and diagonal variance of a Normal distribution
        # self.encoder_z = nn.Sequential(nn.Linear(input_dim + num_class, dim_hidden),
        #                                nn.Softplus(),
        #                                nn.Linear(dim_hidden, dim_hidden),
        #                                # nn.ReLU(),
        #                                # nn.Linear(dim_hidden, dim_hidden),
        #                                nn.Softplus(),
        #                                nn.Linear(dim_hidden, 2 * dim_z),
        #                                torch.nn.BatchNorm1d(dim_z*2, affine=False))
        self.encoder_z = nn.Sequential()
        num_hidden = len(dim_hidden_en_z)
        for _ in range(num_hidden):
            if _==0:
                in_dim=input_dim + num_class
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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        # for p in self.parameters():
        #     p.data.normal_(0, 0.001)
        #     if p.ndimension() == 1: p.data.fill_(0.)
        # for p in self.parameters():
        #     p.data.normal_(0, 0.001)
        #     if p.ndimension() == 1: p.data.fill_(0.)

    # q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x))) -- SSL paper eq 4
    def encode_z(self, x, y):
        xy = torch.cat([x, y], dim=1)
        # print(xy)
        mu, logsigma = self.encoder_z(xy).chunk(2, dim=-1)
        # print(mu)
        # print(logsigma)
        # print(logsigma )
        # print(logsigma.exp())
        return DT.Normal(mu, nn.Softplus()(logsigma).exp())
        # return DT.Normal(mu, logsigma.exp())

    # q(y|x) = Categorical(y|pi_phi(x)) -- SSL paper eq 4
    def encode_y(self, x):
        return DT.OneHotCategorical(logits=self.encoder_y(x))

    # p(x|y,z) = Bernoulli
    # def decode(self, y, z):
    #     yz = torch.cat([y,z], dim=1)
    #     return DT.Bernoulli(logits=self.decoder(yz))
    def decode(self, y, z):
        yz = torch.cat([y, z], dim=1)
        reconstruction = DT.Bernoulli(logits=self.decoder(yz))
        # print(logsigma)
        # print(logsigma**2)
        return reconstruction
    # classification model q(y|x) using the trained q distribution
    def forward(self, x):
        y_probs = self.encode_y(x).probs
        # return y_probs.max(dim=1)[1]  # return pred labels = argmax
        return y_probs