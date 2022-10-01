#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Contains functions to determine terminal states."""

from typing import Callable

import torch

from offline_mbrl.utils.envs import ENV_CATEGORIES


def hopper_termination_function(observations: torch.Tensor) -> torch.Tensor:
    """Computes whether the observations comes from a terminal state.

    Args:
        observations (torch.Tensor): A batch of observations.

    Returns:
        torch.Tensor: The corresponding terminal signals.
    """
    observations = observations.detach().clone()

    height = observations[:, :, 0]
    angle = observations[:, :, 1]
    not_done = (
        torch.isfinite(observations).all(dim=-1)
        * (observations[:, :, 1:] < 100).all(dim=-1)
        * (height > 0.7)
        * (torch.abs(angle) < 0.2)
    )

    dones = ~not_done
    dones = dones[:, :, None]
    return dones


def half_cheetah_termination_function(observations: torch.Tensor) -> torch.Tensor:
    """Computes whether the observations comes from a terminal state.

    Args:
        observations (torch.Tensor): A batch of observations.

    Returns:
        torch.Tensor: The corresponding terminal signals.
    """
    observations = observations.detach().clone()
    dones = torch.zeros((observations.shape[0], observations.shape[1], 1))
    return dones


def walker2d_termination_function(observations: torch.Tensor) -> torch.Tensor:
    """Computes whether the observations comes from a terminal state.

    Args:
        observations (torch.Tensor): A batch of observations.

    Returns:
        torch.Tensor: The corresponding terminal signals.
    """
    observations = observations.detach().clone()
    height = observations[:, :, 0]
    angle = observations[:, :, 1]
    not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
    dones = ~not_done
    dones = dones[:, :, None]
    return dones


termination_functions = {
    "hopper": hopper_termination_function,
    "half_cheetah": half_cheetah_termination_function,
    "walker2d": walker2d_termination_function,
}


def get_termination_function(env_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Retrieves the termination function for an environment.

    The identity function is used if no termination function could be found.

    Args:
        env_name (str): The environment name.

    Returns:
        Callable: The termination function, mapping observations to a terminal signal.
    """
    for category, envs in ENV_CATEGORIES.items():
        if env_name in envs:
            return termination_functions[category]

    return lambda x: x
