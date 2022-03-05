import torch
import torch.nn as nn
import torch.distributions as DT
import torch.nn.functional as F
from torch.nn import init
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
    def __init__(self, dim_in,num_class,dim_z,dim_hidden,device='cpu'):
        super().__init__()
        if len(dim_in)==2:
            H, W = dim_in
            dim_x = H * W
        elif len(dim_in)==3:
            C, H, W = dim_in
            dim_x = C * H * W
        else:
            dim_x= dim_in

        # --------------------
        # p model -- SSL paper generative semi supervised model M2
        # --------------------

        self.p_y = DT.OneHotCategorical(probs=1 / num_class * torch.ones(1,num_class, device=device))
        self.p_z = DT.Normal(torch.tensor(0., device=device), torch.tensor(1., device=device))

        # parametrized data likelihood p(x|y,z)
        self.decoder = nn.Sequential(nn.Linear(dim_z + num_class, dim_hidden),
                                     nn.ReLU(),
                                     nn.Linear(dim_hidden, dim_hidden),
                                     nn.ReLU(),
                                     nn.Linear(dim_hidden, dim_hidden),
                                     nn.ReLU(),
                                     nn.Linear(dim_hidden, dim_x*2),
                                     torch.nn.BatchNorm1d(dim_x*2, affine=False))

        # --------------------
        # q model -- SSL paper eq 4
        # --------------------

        # parametrized q(y|x) = Cat(y|pi_phi(x)) -- outputs parametrization of categorical distribution
        self.encoder_y = nn.Sequential(nn.Linear(dim_x, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, num_class))

        # parametrized q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x))) -- output parametrizations for mean and diagonal variance of a Normal distribution
        self.encoder_z = nn.Sequential(nn.Linear(dim_x + num_class, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, 2 * dim_z),
                                       torch.nn.BatchNorm1d(dim_z*2, affine=False))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

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

    # q(y|x) = Categorical(y|pi_phi(x)) -- SSL paper eq 4
    def encode_y(self, x):
        return DT.OneHotCategorical(logits=self.encoder_y(x))

    # p(x|y,z) = Bernoulli
    # def decode(self, y, z):
    #     yz = torch.cat([y,z], dim=1)
    #     return DT.Bernoulli(logits=self.decoder(yz))
    def decode(self, y, z):
        yz = torch.cat([y, z], dim=1)
        mu, logsigma = self.decoder(yz).chunk(2, dim=-1)
        # print(logsigma)
        # print(logsigma**2)
        return DT.Normal(mu, nn.Softplus()(logsigma).exp())
    # classification model q(y|x) using the trained q distribution
    def forward(self, x):
        y_probs = self.encode_y(x).probs
        # return y_probs.max(dim=1)[1]  # return pred labels = argmax
        return y_probs