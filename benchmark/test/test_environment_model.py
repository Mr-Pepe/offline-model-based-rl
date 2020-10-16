from torch._C import dtype
from benchmark.models.environment_model import EnvironmentModel
import torch.nn as nn
import torch
from torch.optim.adam import Adam
import d4rl
import gym
import numpy as np


def test_is_nn_module():
    assert issubclass(EnvironmentModel, nn.Module)


def test_takes_state_and_action_as_input_and_outputs_state_and_reward():
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(obs_dim, act_dim)

    tensor_size = (3, obs_dim+act_dim)
    input = torch.rand(tensor_size)
    output = model(input)

    np.testing.assert_array_equal(output.shape, (3, obs_dim+1))


def test_overfits_on_single_sample():
    model = EnvironmentModel(1, 1)

    x = torch.as_tensor([3, 3], dtype=torch.float32)
    y = torch.as_tensor([5, 4], dtype=torch.float32)
    lr = 1e-3

    optim = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    y_pred = 0

    for i in range(1000):
        optim.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        print("Loss: {}".format(loss))
        loss.backward()
        optim.step()

    assert criterion(y, y_pred).item() < 1e-5


def test_overfits_on_batch():

    model = EnvironmentModel(obs_dim=3, act_dim=4)

    x = torch.rand((10, 7))
    y = torch.rand((10, 4))
    lr = 1e-3

    optim = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    y_pred = 0

    for i in range(1000):
        optim.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        print("Loss: {}".format(loss))
        loss.backward()
        optim.step()

    assert criterion(y, y_pred).item() < 1e-5


def test_trains_on_offline_data():
    env = gym.make('halfcheetah-random-v0')
    dataset = env.get_dataset()
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards'].reshape(-1, 1)

    model = EnvironmentModel(
        observations.shape[1], actions.shape[1])

    lr = 1e-2
    batch_size = 1024

    optim = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for i in range(2000):
        idxs = np.random.randint(0, observations.shape[0] - 1, size=batch_size)
        x = torch.as_tensor(np.concatenate(
            (observations[idxs], actions[idxs]), axis=1), dtype=torch.float32)
        y = torch.as_tensor(np.concatenate(
            (observations[idxs+1], rewards[idxs]), axis=1), dtype=torch.float32)
        optim.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        print("Loss: {}".format(loss))
        losses.append(loss.item())
        loss.backward()
        optim.step()

    assert losses[-1] < 1
