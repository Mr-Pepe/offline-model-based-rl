from benchmark.utils.load_dataset import load_dataset_from_env
import numpy as np
from benchmark.utils.preprocessing import \
    get_preprocessing_function, preprocess_antmaze_medium_diverse, preprocess_antmaze_umaze
from benchmark.utils.replay_buffer import ReplayBuffer
import pytest
import torch
import gym
import d4rl  # noqa


@pytest.mark.medium
def test_antmaze_medium_preprocessing():
    n_samples = 100

    torch.manual_seed(0)

    env = gym.make('antmaze-medium-diverse-v0')
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

    preprocessed = preprocess_antmaze_medium_diverse(obs_act.detach().clone())

    np.testing.assert_array_equal(obs_act.shape, preprocessed.shape)

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal,
        obs_act,
        preprocessed)


@pytest.mark.fast
def test_antmaze_umaze_preprocessing():
    n_samples = 20

    torch.manual_seed(0)

    env = gym.make('antmaze-umaze-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    buffer = ReplayBuffer(obs_dim,
                          act_dim,
                          n_samples)

    start_velocities = torch.as_tensor(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    start_states = torch.as_tensor([
        [0.0, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        [2.0, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        [4.0, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        [6.0, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        [8.0, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        [8.0, 2.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        [8.0, 4.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        [8.0, 6.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        [8.0, 8.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        [6.0, 8.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        [4.0, 8.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        [2.0, 8.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        [0.0, 8.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]],)

    env.reset()

    start_state = start_states[torch.randint(0, len(start_states), (1,))][0]
    env.set_state(start_state,
                  start_velocities)

    o = torch.cat((start_state, start_velocities))

    for step in range(n_samples):
        if step % 100 == 0:
            print("{}/{}".format(step, n_samples), end='\r')
        a = env.action_space.sample()
        o2, r, d, _ = env.step(a)
        buffer.store(torch.as_tensor(o),
                     torch.as_tensor(a),
                     torch.as_tensor(r),
                     torch.as_tensor(o2),
                     torch.as_tensor(d))

        if d or (step+1) % 1000 == 0:
            env.reset()
            start_state = start_states[torch.randint(
                0, len(start_states), (1,))][0]
            env.set_state(start_state,
                          start_velocities)
            o = torch.cat((start_state, start_velocities))
        else:
            o = o2

    obs_act = torch.cat((buffer.obs_buf, buffer.act_buf), dim=1)

    preprocessed = preprocess_antmaze_umaze(obs_act.detach().clone())

    np.testing.assert_array_equal(obs_act.shape, preprocessed.shape)

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal,
        obs_act,
        preprocessed)


@pytest.mark.current
@pytest.mark.medium
def test_cheetah_medium_replay_preprocessing():
    n_samples = 100

    torch.manual_seed(0)

    env_name = 'halfcheetah-medium-replay-v1'

    pre_fn = get_preprocessing_function(env_name)
    assert pre_fn is not None

    env = gym.make(env_name)
    buffer, obs_dim, act_dim = load_dataset_from_env(env)

    obs_act = torch.cat((buffer.obs_buf, buffer.act_buf), dim=1)

    preprocessed = pre_fn(obs_act)

    np.testing.assert_array_equal(obs_act.shape, preprocessed.shape)

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal,
        obs_act,
        preprocessed)

    assert preprocessed.mean(dim=0).abs().sum() < 0.1
    assert (1 - preprocessed.std(dim=0)).abs().sum() < 0.1
