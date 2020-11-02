from benchmark.utils.termination_functions import termination_fn
import gym
import numpy as np


def test_hopper_termination_function():
    env = gym.make('Hopper-v2')
    env.reset()

    next_obss = []
    dones = []

    for step in range(1000):
        next_obs, _, done, _ = env.step(env.action_space.sample())

        next_obss.append(next_obs)
        dones.append(done)

        if done:
            env.reset()

    next_obss = np.array(next_obss)
    dones = np.array(dones).reshape(-1, 1)

    np.testing.assert_array_equal(dones, termination_fn(next_obss))
