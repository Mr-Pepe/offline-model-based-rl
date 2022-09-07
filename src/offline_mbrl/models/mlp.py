from typing import Type

from torch import nn


def mlp(
    sizes: tuple[int, ...],
    activation: Type[nn.Module],
    output_activation: Type[nn.Module] = nn.Identity,
) -> nn.Sequential:
    # Based on https://spinningup.openai.com

    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)
