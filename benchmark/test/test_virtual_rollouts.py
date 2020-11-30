from benchmark.utils.replay_buffer import ReplayBuffer
import numpy as np
import pytest
import torch
from benchmark.utils.virtual_rollouts import generate_virtual_rollout, \
    generate_virtual_rollouts
from benchmark.actors.sac import SAC
from benchmark.models.environment_model import EnvironmentModel
import gym
from benchmark.utils.termination_functions import termination_functions
import time


@pytest.mark.fast
def test_generate_rollout_of_desired_length():
    env = gym.make('HalfCheetah-v2')
    observation_space = env.observation_space
    action_space = env.action_space

    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    start_observation = torch.as_tensor(env.reset(),
                                        dtype=torch.float32).unsqueeze(0)

    buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=3)
    buffer.store(start_observation, 0, 0, 0, 0)

    model = EnvironmentModel(obs_dim, act_dim, type='probabilistic')
    agent = SAC(observation_space, action_space)

    virtual_rollout = generate_virtual_rollouts(
        model,
        agent,
        buffer,
        10,
        n_rollouts=1,
        stop_on_terminal=False,
        term_fn=termination_functions['half_cheetah'])

    assert len(virtual_rollout['obs']) == 10

    np.testing.assert_array_equal(virtual_rollout['obs'].shape, (10, 17))
    np.testing.assert_array_equal(virtual_rollout['act'].shape, (10, 6))
    np.testing.assert_array_equal(virtual_rollout['rew'].shape, (10))
    np.testing.assert_array_equal(virtual_rollout['next_obs'].shape, (10, 17))
    np.testing.assert_array_equal(virtual_rollout['done'].shape, (10))


@pytest.mark.fast
def test_generate_rollout_stops_on_terminal():
    torch.manual_seed(0)
    env = gym.make('HalfCheetah-v2')
    observation_space = env.observation_space
    action_space = env.action_space

    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    start_observation = torch.as_tensor(env.reset(),
                                        dtype=torch.float32).unsqueeze(0)

    buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=3)
    buffer.store(start_observation, 0, 0, 0, 0)

    model = EnvironmentModel(obs_dim, act_dim, type='probabilistic')
    agent = SAC(observation_space, action_space)

    virtual_rollout = generate_virtual_rollouts(
        model,
        agent,
        buffer,
        10,
        stop_on_terminal=True,
        term_fn=lambda x: torch.ones((1, 1, 1))
    )

    assert len(virtual_rollout['obs']) < 10


@pytest.mark.medium
def test_generating_and_saving_rollouts_in_parallel_is_faster():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make('maze2d-open-dense-v0')
    observation_space = env.observation_space
    action_space = env.action_space

    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    start_observation = torch.as_tensor(env.reset(),
                                        dtype=torch.float32,
                                        device=device).unsqueeze(0)

    parallel_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=int(1e6), device=device)
    parallel_buffer.store(start_observation, 0, 0, 0, 0)

    sequential_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=int(1e6), device=device)
    sequential_buffer.store(start_observation, 0, 0, 0, 0)

    model = EnvironmentModel(obs_dim, act_dim, type='probabilistic')
    model.to(device)
    agent = SAC(observation_space, action_space, device=device)

    n_runs = 5
    n_rollouts = 100
    rollout_length = 3

    start_time = time.time()
    for i in range(n_runs):
        rollout = generate_virtual_rollouts(
            model, agent, parallel_buffer, rollout_length,
            n_rollouts=n_rollouts,
            stop_on_terminal=False)

        parallel_buffer.store_batch(rollout['obs'],
                                    rollout['act'],
                                    rollout['rew'],
                                    rollout['next_obs'],
                                    rollout['done'])

    time_parallel = time.time() - start_time

    start_time = time.time()

    for i in range(n_runs*n_rollouts):
        rollout = generate_virtual_rollout(model,
                                           agent,
                                           start_observation,
                                           rollout_length,
                                           stop_on_terminal=False)

        for step in rollout:
            sequential_buffer.store(
                step['obs'], step['act'], step['rew'],
                step['next_obs'], step['done'])

    time_sequential = time.time() - start_time

    assert parallel_buffer.size == sequential_buffer.size

    print("Parallel: {:.3f}s Sequential: {:.3f}s".format(time_parallel,
                                                         time_sequential))

    assert time_parallel < time_sequential


@pytest.mark.medium
def test_use_random_actions_in_virtual_rollout():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make('maze2d-open-dense-v0')
    observation_space = env.observation_space
    action_space = env.action_space

    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    start_observation = torch.as_tensor(env.reset(),
                                        dtype=torch.float32,
                                        device=device).unsqueeze(0)

    buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=int(1e6), device=device)
    buffer.store(start_observation, 0, 0, 0, 0)

    model = EnvironmentModel(obs_dim, act_dim, type='probabilistic')
    model.to(device)
    agent = SAC(observation_space, action_space, device=device)

    torch.random.manual_seed(0)
    rollouts1 = generate_virtual_rollouts(model, agent, buffer, 1, 100)
    torch.random.manual_seed(0)
    rollouts2 = generate_virtual_rollouts(model, agent, buffer, 1, 100)
    torch.random.manual_seed(0)
    rollouts3 = generate_virtual_rollouts(model, agent, buffer, 1, 100,
                                          random_action=True)
    torch.random.manual_seed(0)
    rollouts4 = generate_virtual_rollouts(model, agent, buffer, 1, 100,
                                          random_action=True)

    np.testing.assert_array_equal(rollouts1['next_obs'].cpu(),
                                  rollouts2['next_obs'].cpu())
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal,
        rollouts3['next_obs'].cpu(),
        rollouts4['next_obs'].cpu(),)
