import numpy as np
import torch
from benchmark.utils.virtual_rollouts import generate_virtual_rollout
from benchmark.actors.sac import SAC
from benchmark.models.environment_model import EnvironmentModel
import gym


def test_generate_rollout_of_desired_length():
    env = gym.make('HalfCheetah-v2')
    observation_space = env.observation_space
    action_space = env.action_space

    observation_dim = observation_space.shape[0]
    action_dim = action_space.shape[0]

    start_observation = torch.as_tensor(env.reset(), dtype=torch.float32).unsqueeze(0)

    model = EnvironmentModel(observation_dim, action_dim)
    agent = SAC(observation_space, action_space)

    virtual_rollout = generate_virtual_rollout(
        model, agent, start_observation, 10)

    assert len(virtual_rollout) == 10

    np.testing.assert_array_equal(virtual_rollout[0]['o'].shape, (1, 17))
    np.testing.assert_array_equal(virtual_rollout[0]['act'].shape, (1, 6))
    np.testing.assert_array_equal(virtual_rollout[0]['rew'].shape, (1))
    np.testing.assert_array_equal(virtual_rollout[0]['o2'].shape, (1, 17))
    assert virtual_rollout[0]['d'] == False
