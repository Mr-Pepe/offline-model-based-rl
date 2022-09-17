# pylint: disable=unused-import
import gym
import numpy as np
import pytest
import torch

from offline_mbrl.utils.envs import (
    HALF_CHEETAH_RANDOM_V2,
    HOPPER_RANDOM_V2,
    WALKER_RANDOM_V2,
)
from offline_mbrl.utils.termination_functions import (
    get_termination_function,
    half_cheetah_termination_function,
    hopper_termination_function,
    walker2d_termination_function,
)


def run_env(env: gym.Env, n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    env.reset()

    next_obss = []
    dones = []

    for _ in range(n_steps):
        next_obs, _, done, _ = env.step(env.action_space.sample())

        next_obss.append(next_obs)
        dones.append(done)

        if done:
            env.reset()

    return torch.as_tensor(next_obss), torch.as_tensor(dones).reshape(-1, 1)


@pytest.mark.fast
def test_hopper_postprocessing() -> None:
    next_observations, dones = run_env(gym.make(HOPPER_RANDOM_V2), 100)

    assert dones.sum() > 0

    np.testing.assert_array_equal(
        torch.stack(3 * [dones]),
        hopper_termination_function(observations=torch.stack(3 * [next_observations])),
    )


@pytest.mark.fast
def test_half_cheetah_postprocessing() -> None:
    next_observations, dones = run_env(gym.make(HALF_CHEETAH_RANDOM_V2), 100)

    # Halg cheetah does not generate terminal states
    assert dones.sum() == 0

    np.testing.assert_array_equal(
        torch.stack(3 * [dones]),
        half_cheetah_termination_function(
            observations=torch.stack(3 * [next_observations])
        ),
    )


@pytest.mark.fast
def test_walker2d_postprocessing() -> None:

    next_observations, dones = run_env(gym.make(WALKER_RANDOM_V2), 100)

    assert dones.sum() > 0

    np.testing.assert_array_equal(
        torch.stack(3 * [dones]),
        walker2d_termination_function(
            observations=torch.stack(3 * [next_observations])
        ),
    )


@pytest.mark.fast
def test_raises_error_if_no_post_processing_function_found() -> None:
    with pytest.raises(
        ValueError, match="No postprocessing function found for environment 'abc'."
    ):
        get_termination_function("abc")
