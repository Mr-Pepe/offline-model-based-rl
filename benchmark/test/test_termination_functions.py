from benchmark.utils.mazes import plot_antmaze_umaze_walls
import matplotlib.pyplot as plt
import pytest
import numpy as np
import torch
import gym
from benchmark.utils.termination_functions import \
    half_cheetah_termination_fn, \
    hopper_termination_fn, antmaze_umaze_termination_fn, \
    walker2d_termination_fn
import d4rl # noqa


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

    np.testing.assert_array_equal(torch.stack(3*[dones]),
                                  hopper_termination_fn(
                                      torch.stack(3*[next_observations])))


@pytest.mark.fast
def test_half_cheetah_termination_function():
    next_observations, dones = run_env(gym.make('HalfCheetah-v2'), 100)

    # Halg cheetah does not generate terminal states
    assert dones.sum() == 0

    np.testing.assert_array_equal(torch.stack(3*[dones]),
                                  half_cheetah_termination_fn(
                                      torch.stack(3*[next_observations])))


@pytest.mark.fast
def test_walker2d_termination_function():

    next_observations, dones = run_env(gym.make('Walker2d-v2'), 100)

    assert dones.sum() > 0

    np.testing.assert_array_equal(torch.stack(3*[dones]),
                                  walker2d_termination_fn(
                                      torch.stack(3*[next_observations])))


@pytest.mark.medium
def test_umaze_termination_function():

    env = gym.make('antmaze-umaze-v0')

    x_points = 200
    y_points = 200

    observations = torch.zeros((x_points*y_points,
                                env.observation_space.shape[0]))

    torch.manual_seed(0)
    x_random = torch.rand((x_points,))*30-10
    y_random = torch.rand((y_points,))*30-10

    for i_x, x in enumerate(x_random):
        for i_y, y in enumerate(y_random):
            observations[i_x*i_y, 0] = x
            observations[i_x*i_y, 1] = y

    dones = antmaze_umaze_termination_fn(observations.unsqueeze(0)).view(-1)

    np.testing.assert_array_equal(torch.stack(3*[dones]).shape,
                                  antmaze_umaze_termination_fn(
                                      torch.stack(3*[observations])).shape[:2])

    # plot_umaze_walls([-20, 20], [-20, 20])
    # plt.scatter(observations[dones == False, 0],
    #             observations[dones == False, 1], zorder=1)
    # plt.scatter(observations[dones == True, 0],
    #             observations[dones == True, 1], zorder=2)
    # plt.show()
