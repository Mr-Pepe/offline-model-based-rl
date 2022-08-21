import d4rl  # pylint: disable=unused-import
import gym
import numpy as np
import pytest
import torch

from offline_mbrl.utils.envs import HOPPER_ORIGINAL, WALKER_ORIGINAL
from offline_mbrl.utils.postprocessing import (
    postprocess_half_cheetah,
    postprocess_hopper,
    postprocess_walker2d,
)


def run_env(env, n_steps):
    env.reset()

    next_obss = []
    dones = []

    for step in range(n_steps):
        next_obs, _, done, _ = env.step(env.action_space.sample())

        next_obss.append(next_obs)
        dones.append(done)

        if done:
            env.reset()

    next_obss = torch.as_tensor(next_obss)
    dones = torch.as_tensor(dones).reshape(-1, 1)

    return next_obss, dones


@pytest.mark.fast
def test_hopper_postprocessing():
    next_observations, dones = run_env(gym.make(HOPPER_ORIGINAL), 100)

    assert dones.sum() > 0

    np.testing.assert_array_equal(
        torch.stack(3 * [dones]),
        postprocess_hopper(next_obs=torch.stack(3 * [next_observations]))["dones"],
    )


@pytest.mark.fast
def test_half_cheetah_postprocessing():
    next_observations, dones = run_env(gym.make("HalfCheetah-v2"), 100)

    # Halg cheetah does not generate terminal states
    assert dones.sum() == 0

    np.testing.assert_array_equal(
        torch.stack(3 * [dones]),
        postprocess_half_cheetah(next_obs=torch.stack(3 * [next_observations]))[
            "dones"
        ],
    )


@pytest.mark.fast
def test_walker2d_postprocessing():

    next_observations, dones = run_env(gym.make(WALKER_ORIGINAL), 100)

    assert dones.sum() > 0

    np.testing.assert_array_equal(
        torch.stack(3 * [dones]),
        postprocess_walker2d(next_obs=torch.stack(3 * [next_observations]))["dones"],
    )
