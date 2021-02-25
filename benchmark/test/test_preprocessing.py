from benchmark.utils.envs import HALF_CHEETAH_EXPERT, HALF_CHEETAH_EXPERT_V1, HALF_CHEETAH_MEDIUM, HALF_CHEETAH_MEDIUM_EXPERT, HALF_CHEETAH_MEDIUM_EXPERT_V1, HALF_CHEETAH_MEDIUM_REPLAY, HALF_CHEETAH_MEDIUM_REPLAY_V1, HALF_CHEETAH_MEDIUM_V1, HALF_CHEETAH_RANDOM, HALF_CHEETAH_RANDOM_V1, HOPPER_EXPERT, HOPPER_EXPERT_V1, HOPPER_MEDIUM, HOPPER_MEDIUM_EXPERT, HOPPER_MEDIUM_EXPERT_V1, HOPPER_MEDIUM_REPLAY, HOPPER_MEDIUM_REPLAY_V1, HOPPER_MEDIUM_V1, HOPPER_RANDOM, HOPPER_RANDOM_V1, WALKER_EXPERT, WALKER_EXPERT_v1, WALKER_MEDIUM, WALKER_MEDIUM_EXPERT, WALKER_MEDIUM_EXPERT_V1, WALKER_MEDIUM_REPLAY, WALKER_MEDIUM_REPLAY_V1, WALKER_MEDIUM_v1, WALKER_RANDOM, WALKER_RANDOM_v1
from benchmark.utils.load_dataset import load_dataset_from_env
import numpy as np
from benchmark.utils.preprocessing import \
    get_preprocessing_function, envs_with_preprocessing_functions
from benchmark.utils.replay_buffer import ReplayBuffer
import pytest
import torch
import gym
import d4rl  # noqa


@pytest.mark.current
@pytest.mark.slow
def test_preprocessing():
    for env_name in envs_with_preprocessing_functions:
        print(env_name)
        torch.manual_seed(0)

        pre_fn = get_preprocessing_function(env_name)
        assert pre_fn is not None

        env = gym.make(env_name)
        dataset = env.get_dataset()

        obs_act = torch.cat((torch.as_tensor(dataset['observations']), torch.as_tensor(dataset['actions'])), dim=1)

        preprocessed = pre_fn(obs_act)

        np.testing.assert_array_equal(obs_act.shape, preprocessed.shape)

        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            obs_act,
            preprocessed)

        assert preprocessed.mean(dim=0).abs().sum() < 0.15
        assert (1 - preprocessed.std(dim=0)).abs().sum() < 0.1

