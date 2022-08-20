import pytest
from offline_mbrl.utils.envs import get_test_env
from offline_mbrl.utils.mazes import (
    ANTMAZE_MEDIUM_DIVERSE_GOAL,
    ANTMAZE_UMAZE_DIVERSE_GOAL,
    ANTMAZE_UMAZE_GOAL,
)


@pytest.mark.fast
def test_antmaze_umaze_test_env():
    env_name = "antmaze-umaze-v0"

    test_env = get_test_env(env_name)

    assert test_env.target_goal == ANTMAZE_UMAZE_GOAL


@pytest.mark.fast
def test_antmaze_umaze_diverse_test_env():
    env_name = "antmaze-umaze-diverse-v0"

    test_env = get_test_env(env_name)

    assert test_env.target_goal == ANTMAZE_UMAZE_DIVERSE_GOAL


@pytest.mark.fast
def test_antmaze_medium_diverse_test_env():
    env_name = "antmaze-medium-diverse-v0"

    test_env = get_test_env(env_name)

    assert test_env.target_goal == ANTMAZE_MEDIUM_DIVERSE_GOAL
