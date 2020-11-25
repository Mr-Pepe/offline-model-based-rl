from benchmark.utils.termination_functions import antmaze_termination_fn, half_cheetah_termination_fn, \
    hopper_termination_fn, walker2d_termination_fn
import gym
import torch
import numpy as np
import pytest


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
def test_hopper_termination_function():
    next_observations, dones = run_env(gym.make('Hopper-v2'), 100)

    assert dones.sum() > 0

    np.testing.assert_array_equal(dones,
                                  hopper_termination_fn(next_observations))


@pytest.mark.fast
def test_half_cheetah_termination_function():
    next_observations, dones = run_env(gym.make('HalfCheetah-v2'), 100)

    # Halg cheetah does not generate terminal states
    assert dones.sum() == 0

    np.testing.assert_array_equal(dones,
                                  half_cheetah_termination_fn(
                                      next_observations))


@pytest.mark.fast
def test_walker2d_termination_function():

    next_observations, dones = run_env(gym.make('Walker2d-v2'), 100)

    assert dones.sum() > 0

    np.testing.assert_array_equal(dones,
                                  walker2d_termination_fn(next_observations))


@pytest.mark.fast
def test_antmaze_termination_function():

    next_observations, dones = run_env(gym.make('antmaze-umaze-diverse-v0'),
                                       100)

    assert antmaze_termination_fn(next_observations).sum() == 0
