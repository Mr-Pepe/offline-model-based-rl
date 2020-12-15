from benchmark.utils.reward_functions import get_reward_function
import pytest
import gym
import d4rl  # noqa
from benchmark.utils.load_dataset import load_dataset_from_env
import matplotlib.pyplot as plt  # noqa


@pytest.mark.medium
def test_antmaze_umaze_reward_function():
    env_name = 'antmaze-umaze-v0'
    env = gym.make(env_name)
    buffer, obs_dim, act_dim = load_dataset_from_env(env, 1000)

    rew_fn = get_reward_function(env_name)

    assert rew_fn is not None

    rewards = rew_fn(buffer.obs_buf.unsqueeze(0))

    assert rewards.shape[0] == 1
    assert rewards.shape[1] == buffer.obs_buf.shape[0]
    assert rewards.shape[2] == 1

    for r in rewards[0]:
        assert r == 0 or r == 1