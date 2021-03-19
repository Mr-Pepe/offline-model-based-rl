from benchmark.models.ensemble_dense_layer import EnsembleDenseLayer
import torch
import torch.nn as nn


class QEnsemble(nn.Module):
    # Based on https://spinningup.openai.com

    def __init__(self, obs_dim, act_dim, n_networks=7):
        super().__init__()

        self.n_networks = n_networks

        self.layers = nn.Sequential(
            EnsembleDenseLayer(obs_dim + act_dim,
                               64,
                               n_networks),
            EnsembleDenseLayer(64,
                               64,
                               n_networks),
            EnsembleDenseLayer(64,
                               64,
                               n_networks),
            EnsembleDenseLayer(64,
                               1,
                               n_networks,
                               non_linearity='linear')
        )

    def forward(self, obs, act):
        input = torch.stack(self.n_networks * [torch.cat([obs, act], dim=-1)])
        q = self.layers(input)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.
