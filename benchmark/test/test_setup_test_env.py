from benchmark.utils.setup_test_env import ANTMAZE_MEDIUM_DIVERSE_START_STATES, setup_test_env
import pytest
import gym
import d4rl # noqa

@pytest.mark.current
@pytest.mark.fast
def test_setup_antmaze_medium_diverse():
    env = gym.make('antmaze-medium-diverse-v0')

    for _ in range(10):

        o = setup_test_env(env)

        assert tuple(o[:2]) in ANTMAZE_MEDIUM_DIVERSE_START_STATES
