import d4rl  # pylint: disable=unused-import
import gym
import numpy as np
import pytest
import torch

from offline_mbrl.utils.preprocessing import (
    envs_with_preprocessing_functions,
    get_preprocessing_function,
)


@pytest.mark.slow
def test_preprocessing():
    for env_name in envs_with_preprocessing_functions:
        print(env_name)
        torch.manual_seed(0)

        pre_fn = get_preprocessing_function(env_name)
        assert pre_fn is not None

        env = gym.make(env_name)
        dataset = env.get_dataset()

        obs_act = torch.cat(
            (
                torch.as_tensor(dataset["observations"]),
                torch.as_tensor(dataset["actions"]),
            ),
            dim=1,
        )

        preprocessed = pre_fn(obs_act)

        np.testing.assert_array_equal(obs_act.shape, preprocessed.shape)

        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, obs_act, preprocessed
        )

        assert preprocessed.mean(dim=0).abs().sum() < 0.15
        assert (1 - preprocessed.std(dim=0)).abs().sum() < 0.1
