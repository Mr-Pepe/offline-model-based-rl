from typing import Type

import torch
from torch import nn

from offline_mbrl.models.mlp import mlp


class MLPQFunction(nn.Module):
    # Based on https://spinningup.openai.com

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...],
        activation: Type[nn.Module],
    ) -> None:
        super().__init__()
        self.q = mlp((obs_dim + act_dim, *hidden_sizes, 1), activation)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q = self.q(torch.cat([observation, action], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.
