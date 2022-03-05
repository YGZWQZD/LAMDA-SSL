import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class Encoder(torch.nn.Module):
    def __init__(self, d_in, d_out, activation_type,
                 train_bn_scaling, noise_level,device='cpu'):
        super(Encoder, self).__init__()
        if len(d_in)==2:
            H, W = d_in
            d_in = H * W
        elif len(d_in)==3:
            C, H, W = d_in
            d_in = C * H * W
        else:
            d_in=d_in
        self.d_in=d_in
        self.d_out = d_out
        self.activation_type = activation_type
        self.train_bn_scaling = train_bn_scaling
        self.noise_level = noise_level
        self.device=device

        # Encoder
        # Encoder only uses W matrix, no bias
        self.linear = torch.nn.Linear(d_in, d_out, bias=False)
        self.linear.weight.data = torch.randn(self.linear.weight.data.size()) / np.sqrt(d_in)

        # Batch Normalization
        # For Relu Beta of batch-norm is redundant, hence only Gamma is trained
        # For Softmax Beta, Gamma are trained
        # batch-normalization bias
        self.bn_normalize_clean = torch.nn.BatchNorm1d(d_out, affine=False)
        self.bn_normalize = torch.nn.BatchNorm1d(d_out, affine=False)
        # if self.use_cuda:
        #     self.bn_beta = Parameter(torch.cuda.FloatTensor(1, d_out))
        # else:
        self.bn_beta = Parameter(torch.FloatTensor(1, d_out).to(self.device))
        self.bn_beta.data.zero_()
        if self.train_bn_scaling:
            # batch-normalization scaling
            # if self.use_cuda:
            #     self.bn_gamma = Parameter(torch.cuda.FloatTensor(1, d_out))
            #     self.bn_gamma.data = torch.ones(self.bn_gamma.size()).cuda()
            # else:
            #     self.bn_gamma = Parameter(torch.FloatTensor(1, d_out))
            #     self.bn_gamma.data = torch.ones(self.bn_gamma.size())
            self.bn_gamma = Parameter(torch.FloatTensor(1, d_out).to(self.device))
            self.bn_gamma.data = torch.ones(self.bn_gamma.size()).to(self.device)

        # Activation
        if activation_type == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation_type == 'softmax':
            self.activation = torch.nn.Softmax()
        else:
            raise ValueError("invalid Acitvation type")

        # buffer for z_pre, z which will be used in decoder cost
        self.buffer_z_pre = None
        self.buffer_z = None
        # buffer for tilde_z which will be used by decoder for reconstruction
        self.buffer_tilde_z = None

    def bn_gamma_beta(self, x):
        # if self.use_cuda:
        #     ones = Parameter(torch.ones(x.size()[0], 1).cuda())
        # else:
        #     ones = Parameter(torch.ones(x.size()[0], 1))
        ones = Parameter(torch.ones(x.size()[0], 1).to(self.device))
        t = x + ones.mm(self.bn_beta)
        if self.train_bn_scaling:
            t = torch.mul(t, ones.mm(self.bn_gamma))
        return t

    def forward_clean(self, h):
        z_pre = self.linear(h)
        # Store z_pre, z to be used in calculation of reconstruction cost
        self.buffer_z_pre = z_pre.detach().clone()
        z = self.bn_normalize_clean(z_pre)
        self.buffer_z = z.detach().clone()
        z_gb = self.bn_gamma_beta(z)
        h = self.activation(z_gb)
        return h

    def forward_noise(self, tilde_h):
        # z_pre will be used in the decoder cost
        z_pre = self.linear(tilde_h)
        z_pre_norm = self.bn_normalize(z_pre)
        # Add noise
        noise = np.random.normal(loc=0.0, scale=self.noise_level, size=z_pre_norm.size())
        # if self.use_cuda:
        #     noise = Variable(torch.cuda.FloatTensor(noise))
        # else:
        noise = Variable(torch.FloatTensor(noise).to(self.device))
        # tilde_z will be used by decoder for reconstruction
        tilde_z = z_pre_norm + noise
        # store tilde_z in buffer
        self.buffer_tilde_z = tilde_z
        z = self.bn_gamma_beta(tilde_z)
        h = self.activation(z)
        return h


class StackedEncoders(torch.nn.Module):
    def __init__(self, d_in, n_class,d_encoders, activation_types,
                 train_batch_norms, noise_std,device='cpu'):
        super(StackedEncoders, self).__init__()
        self.buffer_tilde_z_bottom = None
        self.encoders_ref = []
        self.encoders = torch.nn.Sequential()
        self.noise_level = noise_std
        n_encoders = len(d_encoders)+1
        self.device=device
        for i in range(n_encoders):
            if i == 0:
                d_input = d_in
            else:
                d_input = d_encoders[i - 1]
            if i==n_encoders-1:
                d_output=n_class
            else:
                d_output = d_encoders[i]
            activation = activation_types[i]
            train_batch_norm = train_batch_norms[i]
            encoder_ref = "encoder_" + str(i)
            encoder = Encoder(d_input, d_output, activation, train_batch_norm, noise_std,device)
            self.encoders_ref.append(encoder_ref)
            self.encoders.add_module(encoder_ref, encoder)

    def forward_clean(self, x):
        h = x
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            h = encoder.forward_clean(h)
        return h

    def forward_noise(self, x):
        noise = np.random.normal(loc=0.0, scale=self.noise_level, size=x.size())
        # if self.use_cuda:
        #     noise = Variable(torch.cuda.FloatTensor(noise))
        # else:
        noise = Variable(torch.FloatTensor(noise).to(self.device))
        h = x + noise
        self.buffer_tilde_z_bottom = h.clone()
        # pass through encoders
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            h = encoder.forward_noise(h)
        return h

    def get_encoders_tilde_z(self, reverse=True):
        tilde_z_layers = []
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            tilde_z = encoder.buffer_tilde_z.clone()
            tilde_z_layers.append(tilde_z)
        if reverse:
            tilde_z_layers.reverse()
        return tilde_z_layers

    def get_encoders_z_pre(self, reverse=True):
        z_pre_layers = []
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            z_pre = encoder.buffer_z_pre.clone()
            z_pre_layers.append(z_pre)
        if reverse:
            z_pre_layers.reverse()
        return z_pre_layers

    def get_encoders_z(self, reverse=True):
        z_layers = []
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            z = encoder.buffer_z.clone()
            z_layers.append(z)
        if reverse:
            z_layers.reverse()
        return z_layers

class Decoder(torch.nn.Module):
    def __init__(self, d_in, d_out,device='cpu'):
        super(Decoder, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.device = device


        self.a1 = Parameter(0. * torch.ones(1, d_in).to(self.device))
        self.a2 = Parameter(1. * torch.ones(1, d_in).to(self.device))
        self.a3 = Parameter(0. * torch.ones(1, d_in).to(self.device))
        self.a4 = Parameter(0. * torch.ones(1, d_in).to(self.device))
        self.a5 = Parameter(0. * torch.ones(1, d_in).to(self.device))

        self.a6 = Parameter(0. * torch.ones(1, d_in).to(self.device))
        self.a7 = Parameter(1. * torch.ones(1, d_in).to(self.device))
        self.a8 = Parameter(0. * torch.ones(1, d_in).to(self.device))
        self.a9 = Parameter(0. * torch.ones(1, d_in).to(self.device))
        self.a10 = Parameter(0. * torch.ones(1, d_in).to(self.device))


        if self.d_out is not None:
            self.V = torch.nn.Linear(d_in, d_out, bias=False)
            self.V.weight.data = torch.randn(self.V.weight.data.size()) / np.sqrt(d_in)
            # batch-normalization for u
            self.bn_normalize = torch.nn.BatchNorm1d(d_out, affine=False)

        # buffer for hat_z_l to be used for cost calculation
        self.buffer_hat_z_l = None

    def g(self, tilde_z_l, u_l):
        # if self.use_cuda:
        #     ones = Parameter(torch.ones(tilde_z_l.size()[0], 1).cuda())
        # else:
        ones = Parameter(torch.ones(tilde_z_l.size()[0], 1).to(self.device))

        b_a1 = ones.mm(self.a1)
        b_a2 = ones.mm(self.a2)
        b_a3 = ones.mm(self.a3)
        b_a4 = ones.mm(self.a4)
        b_a5 = ones.mm(self.a5)

        b_a6 = ones.mm(self.a6)
        b_a7 = ones.mm(self.a7)
        b_a8 = ones.mm(self.a8)
        b_a9 = ones.mm(self.a9)
        b_a10 = ones.mm(self.a10)

        mu_l = torch.mul(b_a1, torch.sigmoid(torch.mul(b_a2, u_l) + b_a3)) + \
               torch.mul(b_a4, u_l) + \
               b_a5

        v_l = torch.mul(b_a6, torch.sigmoid(torch.mul(b_a7, u_l) + b_a8)) + \
              torch.mul(b_a9, u_l) + \
              b_a10

        hat_z_l = torch.mul(tilde_z_l - mu_l, v_l) + mu_l

        return hat_z_l

    def forward(self, tilde_z_l, u_l):
        # hat_z_l will be used for calculating decoder costs
        hat_z_l = self.g(tilde_z_l, u_l)
        # store hat_z_l in buffer for cost calculation
        self.buffer_hat_z_l = hat_z_l

        if self.d_out is not None:
            t = self.V.forward(hat_z_l)
            u_l_below = self.bn_normalize(t)
            return u_l_below
        else:
            return None


class StackedDecoders(torch.nn.Module):
    def __init__(self, d_in, n_class,d_decoders, device='cpu'):
        super(StackedDecoders, self).__init__()
        # print(d_in)
        # print(n_class)
        self.bn_u_top = torch.nn.BatchNorm1d(n_class, affine=False)
        self.decoders_ref = []
        self.decoders = torch.nn.Sequential()
        n_decoders = len(d_decoders)+1
        self.device=device
        for i in range(n_decoders):
            if i == 0:
                d_input = n_class
            else:
                d_input = d_decoders[i - 1]
            if i==n_decoders-1:
                d_output = d_in
            else:
                d_output = d_decoders[i]
            decoder_ref = "decoder_" + str(i)
            decoder = Decoder(d_input, d_output,device=self.device)
            self.decoders_ref.append(decoder_ref)
            self.decoders.add_module(decoder_ref, decoder)

        self.bottom_decoder = Decoder(d_in, None)

    def forward(self, tilde_z_layers, u_top, tilde_z_bottom):
        # Note that tilde_z_layers should be in reversed order of encoders
        hat_z = []
        u = self.bn_u_top(u_top)
        for i in range(len(self.decoders_ref)):
            d_ref = self.decoders_ref[i]
            decoder = getattr(self.decoders, d_ref)
            tilde_z = tilde_z_layers[i]
            u = decoder.forward(tilde_z, u)
            hat_z.append(decoder.buffer_hat_z_l)
        self.bottom_decoder.forward(tilde_z_bottom, u)
        hat_z_bottom = self.bottom_decoder.buffer_hat_z_l.clone()
        hat_z.append(hat_z_bottom)
        return hat_z

    def bn_hat_z_layers(self, hat_z_layers, z_pre_layers):
        # TODO: Calculate batchnorm using GPU Tensors.
        assert len(hat_z_layers) == len(z_pre_layers)
        hat_z_layers_normalized = []
        # print(hat_z_layers.shape)
        # print(z_pre_layers.shape)
        for i, (hat_z, z_pre) in enumerate(zip(hat_z_layers, z_pre_layers)):
            # if self.use_cuda:
            #     ones = Variable(torch.ones(z_pre.size()[0], 1).cuda())
            # else:
            # print(hat_z.shape)# 100*10
            # print(z_pre.shape)# 100*10
            ones = Variable(torch.ones(z_pre.size()[0], 1).to(self.device)) # 10*1
            # print(z_pre.shape)
            mean = torch.mean(z_pre, 0).unsqueeze(0)# 1*10
            # print(mean.shape)

            noise_var = Variable(torch.FloatTensor(np.random.normal(loc=0.0, scale=1 - 1e-10, size=z_pre.size())).to(self.device))
            # if self.use_cuda:
            #     var = np.var(z_pre.data.cpu().numpy() + noise_var, axis=0).reshape(1, z_pre.size()[1])
            # else:
            # var = np.var(z_pre.data.numpy() + noise_var, axis=0).reshape(1, z_pre.size()[1])
            var = torch.var(z_pre.data + noise_var,dim=0).reshape(1, z_pre.size()[1])
            # var = Variable(torch.FloatTensor(var))
            # if self.use_cuda:
            #     hat_z = hat_z.cpu()
            #     ones = ones.cpu()
            #     mean = mean.cpu()
            # print(mean.shape)
            # # print(type(mean))
            # # print(ones.mm(torch.sqrt(var + 1e-10)))
            # print(ones.shape)
            hat_z_normalized = torch.div(hat_z - ones.mm(mean), ones.mm(torch.sqrt(var + 1e-10)))
            # if self.use_cuda:
            #     hat_z_normalized = hat_z_normalized.cuda()
            hat_z_layers_normalized.append(hat_z_normalized)
        return hat_z_layers_normalized

class Ladder(torch.nn.Module):
    def __init__(self, encoder_sizes=[1000, 500, 250, 250, 250], encoder_activations=["relu", "relu", "relu", "relu", "relu", "softmax"],
                 encoder_train_bn_scaling=[False, False, False, False, False, True], noise_std=0.2,dim_in=32*32,n_class=10,device='cpu'):
        super(Ladder, self).__init__()

        decoder_sizes = list(reversed(encoder_sizes))
        # print(encoder_sizes)
        # print(decoder_sizes)
        decoder_in = n_class
        encoder_in = dim_in
        self.device=device
        self.se = StackedEncoders(encoder_in,decoder_in, encoder_sizes, encoder_activations,
                                  encoder_train_bn_scaling, noise_std,device)
        self.de = StackedDecoders(encoder_in,decoder_in, decoder_sizes,device)
        self.bn_image = torch.nn.BatchNorm1d(encoder_in, affine=False)

    def forward_encoders_clean(self, data):
        return self.se.forward_clean(data)

    def forward_encoders_noise(self, data):
        return self.se.forward_noise(data)

    def forward_decoders(self, tilde_z_layers, encoder_output, tilde_z_bottom):
        return self.de.forward(tilde_z_layers, encoder_output, tilde_z_bottom)

    def get_encoders_tilde_z(self, reverse=True):
        return self.se.get_encoders_tilde_z(reverse)

    def get_encoders_z_pre(self, reverse=True):
        return self.se.get_encoders_z_pre(reverse)

    def get_encoder_tilde_z_bottom(self):
        return self.se.buffer_tilde_z_bottom.clone()

    def get_encoders_z(self, reverse=True):
        return self.se.get_encoders_z(reverse)

    def decoder_bn_hat_z_layers(self, hat_z_layers, z_pre_layers):
        return self.de.bn_hat_z_layers(hat_z_layers, z_pre_layers)

    def forward(self, data):
        return self.forward_encoders_clean(data)