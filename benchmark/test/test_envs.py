from benchmark.utils.envs import get_test_env
import pytest


@pytest.mark.fast
def test_antmaze_umaze_test_env():
    env_name = 'antmaze-umaze-v0'

    test_env = get_test_env(env_name)

    assert test_env.target_goal == (0.6, 9.2)
