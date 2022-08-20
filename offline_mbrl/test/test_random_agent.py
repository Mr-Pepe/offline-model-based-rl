import gym
import pytest
from numpy.testing import assert_array_equal, assert_raises
from offline_mbrl.actors.random_agent import RandomAgent
from offline_mbrl.utils.envs import HOPPER_ORIGINAL


@pytest.mark.fast
def test_returns_random_actions():
    env = gym.make(HOPPER_ORIGINAL)
    agent = RandomAgent(env)

    for i in range(10):
        action1 = agent.act()
        action2 = agent.act()

        assert_raises(AssertionError, assert_array_equal, action1, action2)
