import torch
import torch.nn as nn
from offline_mbrl.models.ensemble_dense_layer import EnsembleDenseLayer


class QEnsemble(nn.Module):
    # Based on https://spinningup.openai.com

    def __init__(self, obs_dim, act_dim, n_networks=10):
        super().__init__()

        self.n_networks = n_networks

        self.layers = nn.Sequential(
            EnsembleDenseLayer(obs_dim + act_dim, 64, n_networks),
            EnsembleDenseLayer(64, 64, n_networks),
            EnsembleDenseLayer(64, 64, n_networks),
            EnsembleDenseLayer(64, 1, n_networks, non_linearity="linear"),
        )

    def forward(self, obs, act):
        input = torch.stack(self.n_networks * [torch.cat([obs, act], dim=-1)])
        q = self.layers(input)
        idx = torch.zeros((1,))

        while idx.sum() < 1 or idx.sum() >= self.n_networks:
            idx = torch.rand((self.n_networks,)) > 0.5

        min_rem = torch.min(q[idx, :, :].mean(dim=0), q[~idx, :, :].mean(dim=0))

        return torch.squeeze(min_rem, -1)  # Critical to ensure q has right shape.
