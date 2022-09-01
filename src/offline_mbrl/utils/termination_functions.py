from typing import Callable

import torch

from offline_mbrl.utils.envs import ENV_CATEGORIES


def hopper_termination_function(next_obs: torch.Tensor) -> torch.Tensor:
    next_obs = next_obs.detach().clone()

    height = next_obs[:, :, 0]
    angle = next_obs[:, :, 1]
    not_done = (
        torch.isfinite(next_obs).all(axis=-1)
        * (next_obs[:, :, 1:] < 100).all(axis=-1)
        * (height > 0.7)
        * (torch.abs(angle) < 0.2)
    )

    dones = ~not_done
    dones = dones[:, :, None]
    return dones


def half_cheetah_termination_function(next_obs: torch.Tensor) -> torch.Tensor:
    next_obs = next_obs.detach().clone()
    dones = torch.zeros((next_obs.shape[0], next_obs.shape[1], 1))
    return dones


def walker2d_termination_function(next_obs: torch.Tensor):
    next_obs = next_obs.detach().clone()
    height = next_obs[:, :, 0]
    angle = next_obs[:, :, 1]
    not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
    dones = ~not_done
    dones = dones[:, :, None]
    return dones


termination_functions = {
    "hopper": hopper_termination_function,
    "half_cheetah": half_cheetah_termination_function,
    "walker2d": walker2d_termination_function,
}


def get_termination_function(env_name: str) -> Callable:
    for category, envs in ENV_CATEGORIES.items():
        if env_name in envs:
            return termination_functions[category]

    raise ValueError(f"No postprocessing function found for environment '{env_name}'.")
