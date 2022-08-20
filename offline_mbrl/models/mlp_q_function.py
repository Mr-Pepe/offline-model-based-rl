import torch
import torch.nn as nn
from offline_mbrl.models.mlp import mlp


class MLPQFunction(nn.Module):
    # Based on https://spinningup.openai.com

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.
