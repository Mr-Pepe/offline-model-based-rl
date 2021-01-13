from benchmark.models.ensemble_dense_layer import EnsembleDenseLayer
import torch.nn as nn
import torch


class MultiHeadMlp(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden, n_networks):
        """
            A multi headed network to use in an environment model.
            The output is multiheaded for observations and reward.
            If double_output is true, it will output two values for each output
            value, i.e., mean and logvar.
        """
        super().__init__()

        self.n_networks = n_networks

        layers = []
        layers.append(EnsembleDenseLayer(obs_dim + act_dim,
                                         hidden[0],
                                         n_networks))
        # layers.append(BatchNorm(n_networks))

        for lyr_idx in range(1, len(hidden) - 1):
            layers.append(EnsembleDenseLayer(hidden[lyr_idx-1],
                                             hidden[lyr_idx],
                                             n_networks))
            # layers.append(BatchNorm(n_networks))

        self.layers = nn.Sequential(*layers)

        obs_out_dim = 2*obs_dim

        self.obs_layer = nn.Sequential(
            EnsembleDenseLayer(hidden[-2], hidden[-1], n_networks),
            # BatchNorm(n_networks),
            EnsembleDenseLayer(hidden[-1], obs_out_dim,
                               n_networks, non_linearity='linear')
        )

        reward_out_dim = 2

        self.reward_layer = nn.Sequential(
            EnsembleDenseLayer(hidden[-2], hidden[-1], n_networks),
            # BatchNorm(n_networks),
            EnsembleDenseLayer(
                hidden[-1], reward_out_dim,
                n_networks, non_linearity='linear')
        )

    def forward(self, x):
        features = self.layers(torch.stack(self.n_networks * [x]))

        obs = self.obs_layer(features)
        reward = self.reward_layer(features)

        return obs, reward


class BatchNorm(nn.Module):
    def __init__(self, n_features) -> None:
        super().__init__()

        self.layer = nn.BatchNorm1d(n_features)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)

        out = self.layer(x)
        return torch.transpose(out, 0, 1)
