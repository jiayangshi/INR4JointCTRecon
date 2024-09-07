import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.distributions.kl import kl_divergence

# ref: https://github.com/liyues/NeRP/blob/1c7d0b980246a80f2694daf9e77af00b9b24b7f6/networks.py#L58
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)
    
class SIREN(nn.Module):
    def __init__(self, network_depth, network_width, network_input_size, network_output_size, device='cuda'):
        super(SIREN, self).__init__()

        self.network_depth = network_depth
        self.network_width = network_width
        self.network_input_size = network_input_size
        self.network_output_size = network_output_size
        self.loss = nn.MSELoss()

        num_layers = network_depth
        hidden_dim = network_width
        input_dim = network_input_size
        output_dim = network_output_size

        layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        layers.append(SirenLayer(hidden_dim, output_dim, is_last=True))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        self.device = device

    def forward(self, x):
        out = self.model(x)
        return out
    
    def direct_loss_node(self, output, label):
        # Calculate data likelihood -- L2 loss
        log_likelihood_sum = torch.sum(self.loss(output, label))
        return log_likelihood_sum


class bSiren(nn.Module):
    # bSiren: a baysian wrapped Siren module
    def __init__(self, network_depth, network_width, network_input_size, network_output_size, weight_scale=0, rho_offset=-10, lambda_kl=0.1, device='cuda'):
        super(bSiren, self).__init__()
        self.network_depth = network_depth
        self.network_width = network_width
        self.network_input_size = network_input_size
        self.network_output_size = network_output_size
        self.weight_scale = weight_scale
        self.rho_offset = rho_offset
        self.loss = nn.MSELoss()
        self.device = device
        self.w0 = 30 # factor for siren layer
        self.layer_param_shapes = self.get_layer_param_shapes()
        self.mus = nn.ParameterList()
        self.rhos = nn.ParameterList()
        self.lambda_kl = lambda_kl

        # initialize the parameters, refer to siren paper for details
        b = np.sqrt(6 / self.network_input_size) / self.w0
        cnt = 0
        for shape in self.layer_param_shapes:
            if cnt == 0:
                mu = nn.Parameter(Uniform(-1/self.network_input_size,1/self.network_input_size).sample(shape))
                rho = nn.Parameter(self.rho_offset + torch.zeros(shape)) 
            else:
                mu = nn.Parameter(Uniform(-b,b).sample(shape))
                rho = nn.Parameter(self.rho_offset + torch.zeros(shape)) 
            self.mus.append(mu)
            self.rhos.append(rho)
            cnt += 1

    def get_layer_param_shapes(self):
        # get the shapes of the parameters of each layer
        layer_param_shapes = []
        for i in range(self.network_depth):
            if i == 0:
                W_shape = (self.network_input_size, self.network_width)
                b_shape = (self.network_width,)
            elif i == self.network_depth-1:
                W_shape = (self.network_width, self.network_output_size)
                b_shape = (self.network_output_size,)
            else:
                W_shape = (self.network_width, self.network_width)
                b_shape = (self.network_width,)
            layer_param_shapes.extend([W_shape, b_shape])
        return layer_param_shapes
    
    def transform_gaussian_samples(self, mus, rhos, epsilons):
        # compute softplus for variance
        sigmas = [F.softplus(rho) for rho in rhos]
        samples = []
        for j in range(len(mus)): 
            samples.append(mus[j] + sigmas[j] * epsilons[j])
        return samples
    
    def sample_epsilons(self, param_shapes):
        epsilons = [torch.normal(mean=torch.zeros(shape, device=self.device), 
                                 std=torch.ones(shape, device=self.device)) for shape in param_shapes]
        return epsilons
    
    def net(self, X, layer_params):
        # forward pass through the network with sampled parameters
        layer_input = X
        for i in range(len(layer_params) // 2 - 1):
            h_linear = torch.matmul(layer_input, layer_params[2 * i]).add_(layer_params[2 * i + 1])
            layer_input = torch.sin(self.w0*h_linear)

        output = torch.matmul(layer_input, layer_params[-2]).add_(layer_params[-1])
        output = torch.sigmoid(output)
        return output
    
    def direct_loss_node(self, output=None, label=None):
        # Calculate data likelihood -- L2 loss, without KL divergence
        log_likelihood_sum = torch.sum(self.loss(output, label))
        return log_likelihood_sum
    
    def combined_loss_node(self, output=None, label=None, mus=None, sigmas=None, mus_local=None, sigmas_local=None, factor=None):
        # Calculate data likelihood -- L2 loss, with KL divergence
        log_likelihood_sum = torch.sum(self.loss(output, label))
        # compute KL divergence
        mus_local_tensor = torch.cat([m.view(-1) for m in mus_local])
        sigmas_local_tensor = torch.cat([s.view(-1) for s in sigmas_local])
        mus_tensor = torch.cat([m.view(-1) for m in mus])
        sigmas_tensor = torch.cat([s.view(-1) for s in sigmas])

        # Create distributions assuming 1D Normal distributions
        dist_q = Normal(mus_local_tensor, sigmas_local_tensor)
        dist_p = Normal(mus_tensor, sigmas_tensor)

        # Compute KL divergence
        KL_q_w = torch.sum(kl_divergence(dist_q, dist_p))
        return factor*self.lambda_kl*KL_q_w/np.prod(output.shape) + log_likelihood_sum