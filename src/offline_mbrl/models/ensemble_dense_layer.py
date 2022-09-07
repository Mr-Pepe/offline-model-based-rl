from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


# From: https://github.com/nnaisense/MAX/blob/master/models.py
class EnsembleDenseLayer(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        ensemble_size: int,
        non_linearity: str = "leaky_relu",
    ) -> None:
        """
        linear + activation Layer
        there are `ensemble_size` layers
        computation is done using batch matrix multiplication
        hence forward pass through all models in the ensemble can be done in one call
        weights initialized with xavier normal for leaky relu and linear, xavier uniform
        for swish biases are always initialized to zeros
        Args:
            n_in: size of input vector
            n_out: size of output vector
            ensemble_size: number of models in the ensemble
            non_linearity: 'linear', 'swish' or 'leaky_relu'
        """

        super().__init__()

        weights = torch.zeros(ensemble_size, n_in, n_out).float()
        biases = torch.zeros(ensemble_size, 1, n_out).float()

        for weight in weights:
            if non_linearity == "swish":
                nn.init.xavier_uniform_(weight)
            elif non_linearity == "leaky_relu":
                nn.init.kaiming_normal_(weight)
            elif non_linearity == "tanh":
                nn.init.kaiming_normal_(weight)
            elif non_linearity == "linear":
                nn.init.xavier_normal_(weight)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

        self.non_linearity: Callable

        if non_linearity == "leaky_relu":
            self.non_linearity = F.leaky_relu
        elif non_linearity == "tanh":
            self.non_linearity = torch.tanh
        elif non_linearity == "linear":
            self.non_linearity = nn.Identity()

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        out = torch.baddbmm(self.biases, model_input, self.weights)

        return self.non_linearity(out)
