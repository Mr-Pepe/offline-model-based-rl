from benchmark.utils.replay_buffer import ReplayBuffer
import d4rl  # noqa
import gym
import numpy as np


def test_buffer_returns_percentage_of_terminal_states():
    env = gym.make('hopper-random-v0')
    dataset = d4rl.qlearning_dataset(env)
    observations = dataset['observations']
    next_observations = dataset['next_observations']
    actions = dataset['actions']
    rewards = dataset['rewards'].reshape(-1, 1)
    dones = dataset['terminals'].reshape(-1, 1)

    buffer = ReplayBuffer(len(observations[0]), len(
        actions[0]), len(observations))

    for i in range(len(dataset['observations'])):
        buffer.store(
            observations[i],
            actions[i],
            rewards[i],
            next_observations[i],
            dones[i]
        )

    np.testing.assert_almost_equal(
        buffer.get_terminal_ratio(), dones.sum()/dones.size)
