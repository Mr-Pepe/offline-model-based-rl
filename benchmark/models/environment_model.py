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
                 type='deterministic'):
        """
            type (string): deterministic or probabilistic
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = obs_dim+1

        self.type = type
        if type == 'deterministic':
            self.layers = mlp([obs_dim + act_dim] +
                              hidden + [obs_dim+1], nn.ReLU)
        elif type == 'probabilistic':
            # Probabilistic network outputs are first all means and
            # then all log-variances
            self.layers = mlp([obs_dim + act_dim] +
                              hidden + [self.out_dim*2], nn.ReLU)
        else:
            raise ValueError("Unknown type {}".format(type))

        # Taken from https://github.com/kchua/handful-of-trials/blob/master/
        # dmbrl/modeling/models/BNN.py
        self.max_logvar = Parameter(torch.ones((self.out_dim))/2,
                                    requires_grad=True)
        self.min_logvar = Parameter(torch.ones((self.out_dim))*-10,
                                    requires_grad=True)

    def predict_mean_and_logvar(self, x):
        device = next(self.parameters()).device
        x = self.check_device_and_shape(x, device)

        out = self.layers(x)
        mean, logvar = out[:, self.out_dim:], out[:, :self.out_dim]
        logvar = self.max_logvar - softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + softplus(logvar - self.min_logvar)

        # Only learn a residual
        return mean - torch.cat((x[:, :self.obs_dim],
                                 torch.zeros((x.shape[0], 1),
                                             device=device)),
                                dim=1),\
            logvar, \
            self.max_logvar, \
            self.min_logvar

    def forward(self, x):
        device = next(self.parameters()).device
        x = self.check_device_and_shape(x, device)

        if self.type == 'deterministic':
            # Model only returns the residual
            out = self.layers(x)
            out = out + torch.cat((x[:, :self.obs_dim],
                                   torch.zeros((x.shape[0], 1), device=device)),
                                  dim=1)

        else:
            # Residual and current observation are already sumed up inside
            # the function
            mean, logvar, _, _ = self.predict_mean_and_logvar(x)

            std = torch.exp(0.5*logvar)
            out = torch.normal(mean, std)

        # TODO: Transform angular input to [sin(x), cos(x)]
        return out

    def get_prediction(self, x):
        return self.forward(x).cpu().detach().numpy()

    def check_device_and_shape(self, x, device=None):
        if not device:
            device = next(self.parameters()).device

        x = x.to(device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        return x
