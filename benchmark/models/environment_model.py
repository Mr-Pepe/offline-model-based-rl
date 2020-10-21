from benchmark.models.mlp import mlp
import torch.nn as nn
import torch
from torch.nn.functional import softplus
from torch.nn.parameter import Parameter


class EnvironmentModel(nn.Module):
    """ Takes in a state and action and predicts next state and reward. """

    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden=[128, 128],
                 type='deterministic',
                 n_networks=1):
        """
            type (string): deterministic or probabilistic

            n_networks (int): number of networks in the ensemble
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = obs_dim+1

        self.type = type
        self.n_networks = n_networks

        self.networks = []

        for i_network in range(n_networks):
            if type == 'deterministic':
                self.networks.append(mlp([obs_dim + act_dim] +
                                         hidden + [obs_dim+1], nn.ReLU))
            elif type == 'probabilistic':
                # Probabilistic network outputs are first all means and
                # then all log-variances
                self.networks.append(mlp([obs_dim + act_dim] +
                                         hidden + [self.out_dim*2],
                                         nn.ReLU))
            else:
                raise ValueError("Unknown type {}".format(type))

            # Taken from https://github.com/kchua/handful-of-trials/blob/master/
            # dmbrl/modeling/models/BNN.py
            self.networks[i_network].max_logvar = Parameter(torch.ones(
                (self.out_dim))/2,
                requires_grad=True)
            self.networks[i_network].min_logvar = Parameter(torch.ones(
                (self.out_dim))*-10,
                requires_grad=True)

    def forward(self, x, i_network=0):
        network = self.networks[i_network]

        device = next(network.parameters()).device
        x = self.check_device_and_shape(x, device)

        # TODO: Transform angular input to [sin(x), cos(x)]

        out = network(x)
        mean = 0
        logvar = 0
        max_logvar = 0
        min_logvar = 0

        # The model only learns a residual, so the input has to be added
        if self.type == 'deterministic':
            out += out + torch.cat((x[:, :self.obs_dim],
                                    torch.zeros((x.shape[0], 1),
                                                device=device)),
                                   dim=1)

        elif self.type == 'probabilistic':
            mean = out[:, self.out_dim:] + \
                torch.cat((x[:, :self.obs_dim],
                           torch.zeros((x.shape[0], 1), device=device)),
                          dim=1)

            max_logvar = network.max_logvar
            min_logvar = network.min_logvar

            logvar = out[:, :self.out_dim]
            logvar = max_logvar - softplus(max_logvar - logvar)
            logvar = min_logvar + softplus(logvar - min_logvar)

            std = torch.exp(0.5*logvar)
            out = torch.normal(mean, std)

        return out, \
            mean, \
            logvar, \
            max_logvar, \
            min_logvar

    def get_prediction(self, x, i_network=-1):
        i_network = torch.randint(self.n_networks,
                                  (1,)) if i_network == -1 else i_network

        prediction, _, _, _, _ = self.forward(x, i_network)

        return prediction.cpu().detach().numpy()

    def check_device_and_shape(self, x, device):
        x = x.to(device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        return x
