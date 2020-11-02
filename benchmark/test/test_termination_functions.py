from benchmark.utils.termination_functions import half_cheetah_termination_fn, hopper_termination_fn
import gym
import torch
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

    next_obss = torch.as_tensor(next_obss)
    dones = torch.as_tensor(dones).reshape(-1, 1)

    np.testing.assert_array_equal(dones, hopper_termination_fn(next_obss))


def test_half_cheetah_termination_function_always_returns_false():
    env = gym.make('HalfCheetah-v2')
    env.reset()

    next_obss = []

    for step in range(1000):
        next_obs, _, done, _ = env.step(env.action_space.sample())

        next_obss.append(next_obs)

        if done:
            env.reset()

    next_obss = torch.as_tensor(next_obss)
    for done in half_cheetah_termination_fn(next_obss):
        assert not done
