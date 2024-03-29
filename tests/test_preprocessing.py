# pylint: disable=unused-import
import gym
import numpy as np
import pytest
import torch
from d4rl.offline_env import OfflineEnv

from offline_mbrl.utils.envs import ALL_ENVS
from offline_mbrl.utils.preprocessing import get_preprocessing_function


@pytest.mark.slow
def test_preprocessing() -> None:
    for env_name in ALL_ENVS:
        print(env_name)
        torch.manual_seed(0)

        pre_fn = get_preprocessing_function(env_name)
        assert pre_fn is not None

        env: OfflineEnv = gym.make(env_name)
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


@pytest.mark.fast
def test_retrieving_preprocessing_function_for_unknown_env_does_not_normalize() -> None:

    pre_fn = get_preprocessing_function("abc")

    assert torch.ones(5).equal(pre_fn(torch.ones(5)))
