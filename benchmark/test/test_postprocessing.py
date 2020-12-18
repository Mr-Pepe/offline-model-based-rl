from benchmark.utils.mazes import \
    plot_antmaze_medium, plot_antmaze_umaze, \
    plot_maze2d_umaze
import matplotlib.pyplot as plt
import pytest
import numpy as np
import torch
import gym
from benchmark.utils.postprocessing import \
    postprocess_antmaze_medium, postprocess_half_cheetah, \
    postprocess_hopper, \
    postprocess_antmaze_umaze, \
    postprocess_maze2d_umaze, \
    postprocess_walker2d
import d4rl  # noqa


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
    next_observations, dones = run_env(gym.make('Hopper-v2'), 100)

    assert dones.sum() > 0

    np.testing.assert_array_equal(
        torch.stack(3*[dones]),
        postprocess_hopper(next_obs=torch.stack(3*[next_observations]))['dones'])


@pytest.mark.fast
def test_half_cheetah_postprocessing():
    next_observations, dones = run_env(gym.make('HalfCheetah-v2'), 100)

    # Halg cheetah does not generate terminal states
    assert dones.sum() == 0

    np.testing.assert_array_equal(
        torch.stack(3*[dones]),
        postprocess_half_cheetah(
            next_obs=torch.stack(3*[next_observations]))['dones'])


@pytest.mark.fast
def test_walker2d_postprocessing():

    next_observations, dones = run_env(gym.make('Walker2d-v2'), 100)

    assert dones.sum() > 0

    np.testing.assert_array_equal(
        torch.stack(3*[dones]),
        postprocess_walker2d(
            next_obs=torch.stack(3*[next_observations]))['dones'])


@pytest.mark.medium
def test_antmaze_umaze_postprocessing():

    env = gym.make('antmaze-umaze-v0')

    x_points = 50
    y_points = 50
    n_networks = 3

    next_obs = torch.zeros((n_networks,
                            x_points*y_points,
                            env.observation_space.shape[0]))

    torch.manual_seed(0)

    for i_network in range(n_networks):
        x_random = torch.rand((x_points,))*14-2
        y_random = torch.rand((y_points,))*14-2
        z_random = torch.rand((x_points*y_points,))*1.2

        for i_x, x in enumerate(x_random):
            for i_y, y in enumerate(y_random):
                # next_obs[i_network, i_x*i_y, 0] = x
                # next_obs[i_network, i_x*i_y, 1] = y
                next_obs[i_network, i_x*i_y, 2] = z_random[i_x*i_y]

    # Check collision detection
    obs = torch.ones((next_obs.shape[1], next_obs.shape[2]))

    dones = postprocess_antmaze_umaze(next_obs=next_obs,
                                      obs=obs)['dones']

    # plot_antmaze_umaze([-20, 20], [-20, 20])
    # for i_network in range(n_networks):
    #     plt.scatter(
    #         next_obs[i_network, (dones[i_network] == False).view(-1), 0],
    #         next_obs[i_network, (dones[i_network] == False).view(-1), 1],
    #         zorder=1,
    #         color='blue')
    #     plt.scatter(
    #         next_obs[i_network, (dones[i_network] == True).view(-1), 0],
    #         next_obs[i_network, (dones[i_network] == True).view(-1), 1],
    #         zorder=1,
    #         color='red')

    # plt.show()


# @pytest.mark.current
@pytest.mark.medium
def test_antmaze_medium_postprocessing():

    env = gym.make('antmaze-medium-diverse-v0')

    x_points = 100
    y_points = 100
    n_networks = 3

    next_obs = torch.zeros((n_networks,
                            x_points*y_points,
                            env.observation_space.shape[0]))

    torch.manual_seed(0)

    for i_network in range(n_networks):
        x_random = torch.rand((x_points,))*35-7
        y_random = torch.rand((y_points,))*35-7
        z_random = torch.rand((x_points*y_points,))*0.3 + 0.3

        for i_x, x in enumerate(x_random):
            for i_y, y in enumerate(y_random):
                next_obs[i_network, i_x*i_y, 0] = x
                next_obs[i_network, i_x*i_y, 1] = y
                next_obs[i_network, i_x*i_y, 2] = z_random[i_x*i_y]

    # Check collision detection
    obs = torch.ones((next_obs.shape[1], next_obs.shape[2]))

    dones = postprocess_antmaze_medium(next_obs=next_obs,
                                       obs=obs)['dones']

    # plot_antmaze_medium([-10, 30], [-10, 30])
    # for i_network in range(n_networks):
    #     plt.scatter(
    #         next_obs[i_network, (dones[i_network] == False).view(-1), 0],
    #         next_obs[i_network, (dones[i_network] == False).view(-1), 1],
    #         zorder=1,
    #         color='blue')
    #     plt.scatter(
    #         next_obs[i_network, (dones[i_network] == True).view(-1), 0],
    #         next_obs[i_network, (dones[i_network] == True).view(-1), 1],
    #         zorder=1,
    #         color='red')

    # plt.show()


@pytest.mark.medium
def test_maze2d_umaze_postprocessing():

    env = gym.make('maze2d-umaze-v1')

    x_points = 50
    y_points = 50
    n_networks = 3

    next_obs = torch.zeros((n_networks,
                            x_points*y_points,
                            env.observation_space.shape[0]))

    torch.manual_seed(0)

    for i_network in range(n_networks):
        x_random = torch.rand((x_points,))*7-2
        y_random = torch.rand((y_points,))*7-2

        for i_x, x in enumerate(x_random):
            for i_y, y in enumerate(y_random):
                next_obs[i_network, i_x*i_y, 0] = x
                next_obs[i_network, i_x*i_y, 1] = y

    # Check collision detection
    obs = torch.ones((next_obs.shape[1], next_obs.shape[2]))

    dones = postprocess_maze2d_umaze(next_obs=next_obs,
                                     obs=obs)['dones'].view(-1)

    # plot_maze2d_umaze([-2, 5], [-2, 5])
    # for i_network in range(n_networks):
    #     plt.scatter(next_obs[i_network, :, 0],
    #                 next_obs[i_network, :, 1], zorder=1)
    # plt.show()
