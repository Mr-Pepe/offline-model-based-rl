from benchmark.models.mlp import mlp
import torch.nn as nn
import torch

class EnvironmentModel(nn.Module):
    """ Takes in a state and action and predicts next state and reward. """

    def __init__(self, obs_dim, act_dim, hidden=[128, 128]):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.layers = mlp([obs_dim + act_dim] + hidden + [obs_dim+1], nn.ReLU)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        #TODO: Transform angular input to [sin(x), cos(x)]
        # Only learn a residual of the state
        return self.layers(x) + torch.cat((x[:, :self.obs_dim], torch.zeros((x.shape[0], 1))), dim=1)

    def get_prediction(self, x):
        return self.forward(x).cpu().detach().numpy()