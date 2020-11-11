from benchmark.utils.random_agent import RandomAgent
import gym
from numpy.testing import assert_raises, assert_array_equal
import pytest


@pytest.mark.fast
def test_returns_random_actions():
    env = gym.make('Hopper-v2')
    agent = RandomAgent(env)

    for i in range(10):
        action1 = agent.get_action()
        action2 = agent.get_action()

        assert_raises(AssertionError, assert_array_equal, action1, action2)
