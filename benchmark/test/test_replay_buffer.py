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


def test_buffer_returns_whether_it_contains_a_done_state():
    buffer = ReplayBuffer(1, 1, 1000)

    assert not buffer.has_terminal_state()

    for step in range(1000):
        buffer.store(0, 0, 0, 0, False)

    assert not buffer.has_terminal_state()

    for step in range(1000):
        buffer.store(0, 0, 0, 0, True)

    assert buffer.has_terminal_state()


def test_buffer_returns_batch_with_balanced_terminal_signal():
    buffer = ReplayBuffer(1, 1, 1000)

    # Next check that batch is balanced if buffer is unbalanced
    for step in range(1001):
        buffer.store(0, 0, 0, 0, step % 4 == 0)

    assert buffer.get_terminal_ratio() < 0.5

    batch = buffer.sample_balanced_terminal_batch()

    assert sum(batch['done']) == 0.5 * len(batch['done'])
