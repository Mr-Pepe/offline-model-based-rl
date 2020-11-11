from benchmark.utils.replay_buffer import ReplayBuffer
import d4rl  # noqa
import gym
import numpy as np
import torch
import pytest


@pytest.mark.medium
def test_buffer_returns_percentage_of_terminal_states():
    env = gym.make('hopper-random-v0')
    dataset = d4rl.qlearning_dataset(env)
    observations = dataset['observations']
    next_observations = dataset['next_observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    dones = dataset['terminals']

    buffer = ReplayBuffer(len(observations[0]), len(
        actions[0]), len(observations))

    buffer.store_batch(observations, actions, rewards,
                       next_observations, dones)

    np.testing.assert_almost_equal(
        buffer.get_terminal_ratio(), dones.sum()/dones.size)


@pytest.mark.medium
def test_add_batch_to_buffer():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make('maze2d-open-v0')
    dataset = d4rl.qlearning_dataset(env)
    observations = dataset['observations']
    next_observations = dataset['next_observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    dones = dataset['terminals']

    buffer = ReplayBuffer(len(observations[0]),
                          len(actions[0]),
                          2000000,
                          device=device)

    assert buffer.size == 0

    buffer.store_batch(observations, actions, rewards,
                       next_observations, dones)

    assert buffer.size == 977851
    assert buffer.ptr == 977851


@pytest.mark.medium
def test_store_batch_throws_error_if_buffer_not_empty():
    env = gym.make('maze2d-open-v0')
    dataset = d4rl.qlearning_dataset(env)
    observations = dataset['observations']
    next_observations = dataset['next_observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    dones = dataset['terminals']

    buffer = ReplayBuffer(len(observations[0]),
                          len(actions[0]),
                          1000000)

    buffer.store(observations[0],
                 actions[0],
                 rewards[0],
                 next_observations[0],
                 dones[0],)

    assert buffer.size == 1

    with pytest.raises(RuntimeError):
        buffer.store_batch(observations, actions, rewards,
                           next_observations, dones)


@pytest.mark.medium
def test_store_batch_throws_error_if_buffer_too_small():
    env = gym.make('maze2d-open-v0')
    dataset = d4rl.qlearning_dataset(env)
    observations = dataset['observations']
    next_observations = dataset['next_observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    dones = dataset['terminals']

    buffer = ReplayBuffer(len(observations[0]),
                          len(actions[0]),
                          10)

    with pytest.raises(ValueError):
        buffer.store_batch(observations, actions, rewards,
                           next_observations, dones)


@pytest.mark.fast
def test_buffer_returns_whether_it_contains_a_done_state():
    buffer = ReplayBuffer(1, 1, 100)

    assert not buffer.has_terminal_state()

    for step in range(100):
        buffer.store(0, 0, 0, 0, False)

    assert not buffer.has_terminal_state()

    for step in range(100):
        buffer.store(0, 0, 0, 0, True)

    assert buffer.has_terminal_state()


@pytest.mark.fast
def test_buffer_returns_batch_with_balanced_terminal_signal():
    buffer = ReplayBuffer(1, 1, 1000)

    # Next check that batch is balanced if buffer is unbalanced
    for step in range(1001):
        buffer.store(0, 0, 0, 0, step % 4 == 0)

    assert buffer.get_terminal_ratio() < 0.5

    batch = buffer.sample_balanced_terminal_batch()

    assert sum(batch['done']) == 0.5 * len(batch['done'])
