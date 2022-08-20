import d4rl  # noqa
import gym
import matplotlib.pyplot as plt  # noqa
import pytest
from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.reward_functions import (antmaze_medium_diverse_rew_fn,
                                                 antmaze_umaze_diverse_rew_fn,
                                                 get_reward_function)


@pytest.mark.xfail
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


@pytest.mark.xfail
@pytest.mark.medium
def test_antmaze_umaze_diverse_reward_function():
    env_name = 'antmaze-umaze-diverse-v0'
    env = gym.make(env_name)
    buffer, obs_dim, act_dim = load_dataset_from_env(env, 1000)

    rew_fn = get_reward_function(env_name)

    assert rew_fn is not None
    assert rew_fn is antmaze_umaze_diverse_rew_fn

    rewards = rew_fn(buffer.obs_buf.unsqueeze(0))

    assert rewards.shape[0] == 1
    assert rewards.shape[1] == buffer.obs_buf.shape[0]
    assert rewards.shape[2] == 1


@pytest.mark.xfail
@pytest.mark.medium
def test_antmaze_medium_diverse_reward_function():
    env_name = 'antmaze-medium-diverse-v0'
    env = gym.make(env_name)
    buffer, obs_dim, act_dim = load_dataset_from_env(env, 1000)

    rew_fn = get_reward_function(env_name)

    assert rew_fn is not None
    assert rew_fn is antmaze_medium_diverse_rew_fn

    rewards = rew_fn(buffer.obs_buf.unsqueeze(0))

    assert rewards.shape[0] == 1
    assert rewards.shape[1] == buffer.obs_buf.shape[0]
    assert rewards.shape[2] == 1
