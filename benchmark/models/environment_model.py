from benchmark.models.mlp import mlp
import torch.nn as nn
import torch


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

    def predict_mean_and_logvar(self, x):
        x = self.check_device_and_shape(x)

        out = self.layers(x)
        mean, logvar = out[:, self.out_dim:], out[:, :self.out_dim]

        return mean, logvar

    def forward(self, x):
        device = next(self.parameters()).device
        x = self.check_device_and_shape(x, device)

        if self.type == 'deterministic':
            out = self.layers(x)

        else:
            mean, logvar = self.predict_mean_and_logvar(x)

            # Taken from https://github.com/1Konny/Beta-VAE/blob/master/model.py
            std = logvar.div(2).exp()
            eps = std.data.new(std.size()).normal_()

            out = mean + std*eps

        # TODO: Transform angular input to [sin(x), cos(x)]
        # Only learn a residual of the state
        return out + torch.cat((x[:, :self.obs_dim],
                                torch.zeros((x.shape[0], 1), device=device)),
                               dim=1)

    def get_prediction(self, x):
        return self.forward(x).cpu().detach().numpy()

    def check_device_and_shape(self, x, device=None):
        if not device:
            device = next(self.parameters()).device

        x = x.to(device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        return x
