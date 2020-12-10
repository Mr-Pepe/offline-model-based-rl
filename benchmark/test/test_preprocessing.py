import numpy as np
from benchmark.utils.preprocessing import preprocess_maze2d_umaze
from benchmark.utils.replay_buffer import ReplayBuffer
import pytest
import torch
import gym
import d4rl  # noqa


@pytest.mark.medium
def test_maze2d_umaze_preprocessing():
    n_samples = 3

    torch.manual_seed(0)

    env = gym.make('maze2d-umaze-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    buffer = ReplayBuffer(obs_dim,
                          act_dim,
                          n_samples)

    o = env.reset()

    for step in range(n_samples):
        print("{}/{}".format(step, n_samples), end='\r')
        a = env.action_space.sample()
        o2, r, d, _ = env.step(a)
        buffer.store(torch.as_tensor(o),
                     torch.as_tensor(a),
                     torch.as_tensor(r),
                     torch.as_tensor(o2),
                     torch.as_tensor(d))

        if d or (step+1) % 300 == 0:
            o = env.reset()
        else:
            o = o2

    obs_act = torch.cat((buffer.obs_buf, buffer.act_buf), dim=1)

    preprocessed = preprocess_maze2d_umaze(obs_act.detach().clone())

    np.testing.assert_array_equal(obs_act.shape, preprocessed.shape)

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal,
        obs_act,
        preprocessed)
