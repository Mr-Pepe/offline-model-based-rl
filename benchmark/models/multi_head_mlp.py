import torch.nn as nn


class MultiHeadMlp(nn.Module):

    def __init__(self, sizes, obs_dim, double_output=False):
        """
            A multi headed network to use in an environment model.
            The output is multiheaded for observations and reward.
            If double_output is true, it will output two values for each output
            value, i.e., mean and logvar.
        """
        super().__init__()

        layers = []

        for i_layer in range(len(sizes)-2):
            layers += [nn.Linear(sizes[i_layer], sizes[i_layer+1]), nn.ReLU()]

        self.layers = nn.Sequential(*layers)

        obs_out_dim = obs_dim + double_output*obs_dim

        self.obs_layer = nn.Sequential(
            nn.Linear(sizes[-2], sizes[-1]),
            nn.ReLU(),
            nn.Linear(sizes[-1], obs_out_dim)
        )

        other_out_dim = 2 if double_output else 1

        self.reward_layer = nn.Sequential(
            nn.Linear(sizes[-2], sizes[-1]),
            nn.ReLU(),
            nn.Linear(sizes[-1], other_out_dim)
        )

    def forward(self, x):
        """
        Returns three separate outputs for observation, reward, done signal.
        """
        features = self.layers(x)

        obs = self.obs_layer(features)
        reward = self.reward_layer(features)

        return obs, reward
