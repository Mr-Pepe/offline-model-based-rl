from benchmark.models.ensemble_dense_layer import EnsembleDenseLayer
import torch.nn as nn
import torch


class MultiHeadMlp(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden, n_networks, use_batch_norm=False):
        """
            A multi headed network to use in an environment model.
            The output is multiheaded for observations and reward.
            If double_output is true, it will output two values for each output
            value, i.e., mean and logvar.
        """
        super().__init__()

        self.n_networks = n_networks

        obs_layers = []
        obs_layers.append(EnsembleDenseLayer(obs_dim + act_dim,
                                             hidden[0],
                                             n_networks))
        if use_batch_norm:
            obs_layers.append(BatchNorm(n_networks))

        for lyr_idx in range(1, len(hidden)):
            obs_layers.append(EnsembleDenseLayer(hidden[lyr_idx-1],
                                                 hidden[lyr_idx],
                                                 n_networks))
            if use_batch_norm:
                obs_layers.append(BatchNorm(n_networks))

        obs_layers.append(EnsembleDenseLayer(hidden[-1], obs_dim*2,
                                             n_networks, non_linearity='linear'))

        self.obs_layers = nn.Sequential(*obs_layers)

        self.reward_layers = nn.Sequential(
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
                               2,
                               n_networks,
                               non_linearity='linear')
        )

    def forward(self, x):
        stacked_x = torch.stack(self.n_networks * [x])

        obs = self.obs_layers(stacked_x)
        reward = self.reward_layers(stacked_x)

        return obs, reward


class BatchNorm(nn.Module):
    def __init__(self, n_features) -> None:
        super().__init__()

        self.layer = nn.BatchNorm1d(n_features)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)

        out = self.layer(x)
        return torch.transpose(out, 0, 1)
