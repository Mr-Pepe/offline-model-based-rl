import matplotlib.pyplot as plt  # noqa
import pytest
from benchmark.models.environment_model import EnvironmentModel
import torch.nn as nn
import torch
from torch.optim.adam import Adam
import d4rl  # noqa
import gym
import numpy as np
from math import pi as PI


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = gym.make('halfcheetah-random-v0')
    dataset = env.get_dataset()
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards'].reshape(-1, 1)

    model = EnvironmentModel(
        observations.shape[1], actions.shape[1])

    model.to(device)

    lr = 1e-2
    batch_size = 1024

    optim = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for i in range(2000):
        idxs = np.random.randint(0, observations.shape[0] - 1, size=batch_size)
        x = torch.as_tensor(np.concatenate((observations[idxs],
                                            actions[idxs]),
                                           axis=1),
                            dtype=torch.float32,
                            device=device)
        y = torch.as_tensor(np.concatenate((observations[idxs+1],
                                            rewards[idxs]),
                                           axis=1),
                            dtype=torch.float32,
                            device=device)

        optim.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        print("Loss: {}".format(loss))
        losses.append(loss.item())
        loss.backward()
        optim.step()

    assert losses[-1] < 1


def test_probabilistic_model_returns_different_results_for_same_input():
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(obs_dim, act_dim, type='probabilistic')

    tensor_size = (3, obs_dim+act_dim)
    input = torch.rand(tensor_size)
    output1 = model(input).detach()
    output2 = model(input).detach()

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, output1, output2)


def test_raises_error_if_type_unknown():
    with pytest.raises(ValueError):
        EnvironmentModel(1, 2, type="asdasd")


def test_train_probabilistic_model_on_toy_dataset():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.rand((1000,)) * PI - 2*PI
    x = torch.cat((x, torch.rand((1000,)) * PI + PI)).to(device)
    y = torch.sin(x) + torch.normal(0, 0.225 *
                                    torch.abs(torch.sin(1.5*x + PI/8)))\
        .to(device)

    model = EnvironmentModel(
        1, 0, hidden=[200, 200, 200, 200], type='probabilistic')

    model.to(device)

    lr = 1e-4
    optim = Adam(model.parameters(), lr=lr)

    loss = torch.tensor(0)

    for i in range(500):
        optim.zero_grad()
        mean, logvar, max_logvar, min_logvar = model.predict_mean_and_logvar(
            torch.reshape(x, (-1, 1)))
        inv_var = torch.exp(-logvar)

        mse_loss = (torch.square(mean[:, 0] - y) * inv_var[:, 0]).mean()
        var_loss = logvar.mean()
        var_bound_loss = 0.01*max_logvar.sum() - 0.01*min_logvar.sum()
        loss = mse_loss + var_loss + var_bound_loss

        print("Loss: {:.3f}, MSE: {:.3f}, VAR: {:.3f}, VAR BOUND: {:.3f}"
              .format(loss, mse_loss, var_loss, var_bound_loss))
        loss.backward(retain_graph=True)
        optim.step()

    assert loss.item() < -5

    # x_true = torch.range(-3*PI, 3*PI, 0.01)
    # y_true = torch.sin(x_true)

    # mean, logvar, _, _ = model.predict_mean_and_logvar(
    #     torch.reshape(x_true, (-1, 1)))
    # mean = mean[:, 0].detach().cpu()
    # logvar = logvar[:, 0].detach().cpu()

    # std = torch.exp(0.5*logvar)

    # x = x.cpu()
    # y = y.cpu()

    # plt.fill_between(x_true, mean+std, mean-std, color='lightcoral')
    # plt.scatter(x[800:1200], y[800:1200], color='green', marker='x')
    # plt.plot(x_true, y_true, color='black')
    # plt.plot(x_true, mean, color='red')

    # plt.show()
