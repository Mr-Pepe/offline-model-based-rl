from benchmark.utils.envs import HOPPER_ORIGINAL
from benchmark.actors.random_agent import RandomAgent
import gym
from numpy.testing import assert_raises, assert_array_equal
import pytest


@pytest.mark.fast
def test_returns_random_actions():
    env = gym.make(HOPPER_ORIGINAL)
    agent = RandomAgent(env)

    for i in range(10):
        action1 = agent.act()
        action2 = agent.act()

        assert_raises(AssertionError, assert_array_equal, action1, action2)
