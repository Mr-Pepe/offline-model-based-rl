from benchmark.utils.termination_functions import termination_functions
from benchmark.models.mlp import mlp
from benchmark.models.multi_head_mlp import MultiHeadMlp
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
        # Append reward signal
        self.out_dim = obs_dim+1

        self.type = type
        self.n_networks = n_networks

        if type != 'deterministic' and type != 'probabilistic':
            raise ValueError("Unknown type {}".format(type))

        self.networks = []

        for i_network in range(n_networks):
            self.networks.append(MultiHeadMlp(
                [obs_dim + act_dim] + hidden,
                self.obs_dim,
                double_output=type == 'probabilistic'))

            # Taken from https://github.com/kchua/handful-of-trials/blob/master/
            # dmbrl/modeling/models/BNN.py
            self.networks[i_network].max_logvar = Parameter(torch.ones(
                (self.out_dim))/2,
                requires_grad=True)
            self.networks[i_network].min_logvar = Parameter(torch.ones(
                (self.out_dim))*-10,
                requires_grad=True)

        self.networks = nn.ModuleList(self.networks)

        self.done_network = mlp([self.obs_dim, 32, 32, 32, 1],
                                nn.ReLU,
                                nn.Sigmoid)

    def forward(self, obs_act, i_network=0, term_fn=None):

        if term_fn and term_fn not in termination_functions:
            raise ValueError(
                "Unknown termination function: {}".format(term_fn))

        network = self.networks[i_network]

        device = next(network.parameters()).device
        obs_act = self.check_device_and_shape(obs_act, device)

        # TODO: Transform angular input to [sin(x), cos(x)]

        obs, reward = network(obs_act)

        if self.type == 'deterministic':
            if term_fn:
                done = termination_functions[term_fn](obs)
            else:
                done = self.done_network(obs)

            out = torch.cat((obs, reward, done), dim=1)
        else:
            out = torch.cat((obs[:, :self.obs_dim],
                             reward[:, 0].unsqueeze(1),
                             obs[:, self.obs_dim:],
                             reward[:, 1].unsqueeze(1)), dim=1)
        mean = 0
        logvar = 0
        max_logvar = 0
        min_logvar = 0

        # The model only learns a residual, so the input has to be added
        if self.type == 'deterministic':
            out += torch.cat((obs_act[:, :self.obs_dim],
                              torch.zeros((obs_act.shape[0], 2),
                                          device=device)),
                             dim=1)

        elif self.type == 'probabilistic':
            mean = out[:, :self.out_dim] + \
                torch.cat((obs_act[:, :self.obs_dim],
                           torch.zeros((obs_act.shape[0], 1), device=device)),
                          dim=1)

            max_logvar = network.max_logvar
            min_logvar = network.min_logvar

            logvar = out[:, self.out_dim:]
            logvar = max_logvar - softplus(max_logvar - logvar)
            logvar = min_logvar + softplus(logvar - min_logvar)

            std = torch.exp(0.5*logvar)
            out = torch.normal(mean, std)

            if term_fn:
                done = termination_functions[term_fn](obs)
            else:
                done = self.done_network(out[:, :self.obs_dim])

            out = torch.cat((out, done), dim=1)

        return out, \
            mean, \
            logvar, \
            max_logvar, \
            min_logvar

    def get_prediction(self, x, i_network=-1, term_fn=None):
        i_network = torch.randint(self.n_networks,
                                  (1,)) if i_network == -1 else i_network

        prediction, _, _, _, _ = self.forward(x, i_network, term_fn=term_fn)

        prediction[:, -1] = prediction[:, -1] > 0.5

        return prediction.cpu().detach().numpy()

    def check_device_and_shape(self, x, device):
        x = x.to(device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        return x
