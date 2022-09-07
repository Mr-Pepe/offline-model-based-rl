#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module contains an MLP implementation for use in an environment model."""

import torch
from torch import nn

from offline_mbrl.models.ensemble_dense_layer import EnsembleDenseLayer


class MultiHeadMlp(nn.Module):
    """
    A multi headed network to use in an environment model.
    The output is multiheaded for observations and reward.
    Each output is predicted as mean and variance.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_layer_sizes: tuple[int, ...],
        n_networks: int,
    ) -> None:

        super().__init__()

        self.n_networks = n_networks

        obs_layers = []
        obs_layers.append(
            EnsembleDenseLayer(obs_dim + act_dim, hidden_layer_sizes[0], n_networks)
        )

        for i_hidden_layer in range(1, len(hidden_layer_sizes)):
            obs_layers.append(
                EnsembleDenseLayer(
                    hidden_layer_sizes[i_hidden_layer - 1],
                    hidden_layer_sizes[i_hidden_layer],
                    n_networks,
                )
            )

        obs_layers.append(
            EnsembleDenseLayer(
                hidden_layer_sizes[-1], obs_dim * 2, n_networks, non_linearity="linear"
            )
        )

        self.obs_layers = nn.Sequential(*obs_layers)

        self.reward_layers = nn.Sequential(
            EnsembleDenseLayer(obs_dim + act_dim, 64, n_networks),
            EnsembleDenseLayer(64, 64, n_networks),
            EnsembleDenseLayer(64, 64, n_networks),
            EnsembleDenseLayer(64, 2, n_networks, non_linearity="linear"),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stacked_x = torch.stack(self.n_networks * [x])

        obs = self.obs_layers(stacked_x)
        reward = self.reward_layers(stacked_x)

        return obs, reward
