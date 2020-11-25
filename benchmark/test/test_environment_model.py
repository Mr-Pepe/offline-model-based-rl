from benchmark.utils.get_x_y_from_batch import get_x_y_from_batch
from benchmark.utils.replay_buffer import ReplayBuffer
from benchmark.utils.random_agent import RandomAgent
from benchmark.utils.virtual_rollouts import generate_virtual_rollouts
from benchmark.utils.loss_functions import deterministic_loss, \
    probabilistic_loss
from benchmark.utils.load_dataset import load_dataset_from_env
import pytest
import matplotlib.pyplot as plt
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.termination_functions import termination_functions
import torch.nn as nn
import torch
from torch.optim.adam import Adam
import d4rl  # noqa
import gym
import numpy as np
from math import pi as PI

gym.logger.set_level(40)


@pytest.mark.fast
def test_is_nn_module():
    assert issubclass(EnvironmentModel, nn.Module)


@pytest.mark.fast
def test_takes_state_and_action_as_input_and_outputs_state_reward_done():
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(obs_dim, act_dim, [2, 2])

    tensor_size = (3, obs_dim+act_dim)
    input = torch.rand(tensor_size)
    output, _, _, _, _ = model(input)

    np.testing.assert_array_equal(output.shape, (3, obs_dim+2))


@pytest.mark.medium
def test_single_deterministic_network_overfits_on_single_sample():
    torch.manual_seed(0)
    model = EnvironmentModel(1, 1)

    x = torch.as_tensor([3, 3], dtype=torch.float32)
    y = torch.as_tensor([5, 4, 1], dtype=torch.float32)
    lr = 1e-3

    optim = Adam(model.networks[0].parameters(), lr=lr)
    criterion = nn.MSELoss()

    y_pred = 0

    for i in range(1000):
        optim.zero_grad()
        y_pred, _, _, _, _ = model(x)
        loss = criterion(y_pred[:, :-1].view(-1), y[:-1])
        print("Loss: {}".format(loss))
        loss.backward()
        optim.step()

    assert criterion(y[:-1], y_pred[:, :-1].view(-1)).item() < 2e-5


@pytest.mark.medium
def test_single_deterministic_network_overfits_on_batch():
    model = EnvironmentModel(obs_dim=3, act_dim=4)

    x = torch.rand((10, 7))
    y = torch.rand((10, 5))
    lr = 1e-3

    optim = Adam(model.networks[0].parameters(), lr=lr)
    criterion = nn.MSELoss()

    y_pred = 0

    for i in range(1000):
        optim.zero_grad()
        y_pred, _, _, _, _ = model(x)
        loss = criterion(y_pred[:, :-1], y[:, :-1])
        print("Loss: {}".format(loss))
        loss.backward()
        optim.step()

    assert criterion(y[:, :-1], y_pred[:, :-1]).item() < 1e-5


@pytest.mark.medium
def test_deterministic_model_trains_on_offline_data():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = gym.make('halfcheetah-random-v0')
    dataset = env.get_dataset()
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards'].reshape(-1, 1)
    dones = dataset['terminals'].reshape(-1, 1)

    model = EnvironmentModel(observations.shape[1], actions.shape[1])

    model.to(device)

    lr = 1e-2
    batch_size = 1024

    obs_rew_optim = Adam(model.networks[0].parameters(), lr=lr)
    done_optim = Adam(model.done_network.parameters(), lr=lr)
    obs_rew_criterion = nn.MSELoss()
    done_criterion = nn.BCELoss()

    losses = []

    for i in range(2000):
        idxs = np.random.randint(0, observations.shape[0] - 1, size=batch_size)
        x = torch.as_tensor(np.concatenate((observations[idxs],
                                            actions[idxs]),
                                           axis=1),
                            dtype=torch.float32,
                            device=device)
        y = torch.as_tensor(np.concatenate((observations[idxs+1],
                                            rewards[idxs],
                                            dones[idxs]),
                                           axis=1),
                            dtype=torch.float32,
                            device=device)

        obs_rew_optim.zero_grad()
        done_optim.zero_grad()
        y_pred, _, _, _, _ = model(x)
        obs_rew_loss = obs_rew_criterion(y_pred[:, :-1], y[:, :-1])
        done_loss = done_criterion(y_pred[:, -1], y[:, -1])
        print("Obs/Reward loss: {}   Done loss: {}".format(obs_rew_loss,
                                                           done_loss))
        loss = obs_rew_loss + done_loss
        losses.append(loss.item())
        loss.backward()
        obs_rew_optim.step()
        done_optim.step()

    assert losses[-1] < 1


@pytest.mark.fast
def test_probabilistic_model_returns_different_results_for_same_input():
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(obs_dim, act_dim, [2, 2], type='probabilistic')

    tensor_size = (3, obs_dim+act_dim)
    input = torch.rand(tensor_size)
    output1, _, _, _, _ = model(input)
    output2, _, _, _, _ = model(input)

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal,
        output1.detach(),
        output2.detach())


@pytest.mark.fast
def test_raises_error_if_type_unknown():
    with pytest.raises(ValueError):
        EnvironmentModel(1, 2, [2, 2], type="asdasd")


@pytest.mark.medium
def test_probabilistic_model_trains_on_toy_dataset(steps=3000, plot=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(0)

    x = torch.rand((1000,)) * PI - 2*PI
    x = torch.cat((x, torch.rand((1000,)) * PI + PI)).to(device)
    y = torch.sin(x) + torch.normal(0, 0.225 *
                                    torch.abs(torch.sin(1.5*x + PI/8)))\
        .to(device)

    model = EnvironmentModel(
        1, 0, hidden=[64, 64, 64], type='probabilistic')

    model.to(device)

    lr = 1e-4
    optim = Adam(model.networks[0].parameters(), lr=lr)

    loss = torch.tensor(0)

    for i in range(steps):
        optim.zero_grad()
        _, mean, logvar, max_logvar, min_logvar = model(
            torch.reshape(x, (-1, 1)))
        inv_var = torch.exp(-logvar)

        mse_loss = (torch.square(mean[:, 0] - y) * inv_var[:, 0]).mean()
        var_loss = logvar[:, 0].mean()
        var_bound_loss = 0.01 * \
            max_logvar[0].sum() - 0.01 * min_logvar[0].sum()
        loss = mse_loss + var_loss + var_bound_loss

        print("Loss: {:.3f}, MSE: {:.3f}, VAR: {:.3f}, VAR BOUND: {:.3f}"
              .format(loss, mse_loss, var_loss, var_bound_loss))
        loss.backward(retain_graph=True)
        optim.step()

    if not plot:
        assert loss.item() < -2

    else:
        x_true = torch.arange(-3*PI, 3*PI, 0.01)
        y_true = torch.sin(x_true)

        _, mean, logvar, _, _ = model(torch.reshape(x_true, (-1, 1)))
        mean = mean[:, 0].detach().cpu()
        logvar = logvar[:, 0].detach().cpu()

        std = torch.exp(0.5*logvar)

        x = x.cpu()
        y = y.cpu()

        plt.fill_between(x_true, mean+std, mean-std, color='lightcoral')
        plt.scatter(x[800:1200], y[800:1200], color='green', marker='x')
        plt.plot(x_true, y_true, color='black')
        plt.plot(x_true, mean, color='red')

        plt.show()


@pytest.mark.fast
def test_deterministic_ensemble_gives_different_predictions_per_model():
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(obs_dim, act_dim, n_networks=3)

    tensor_size = (3, obs_dim+act_dim)
    input = torch.rand(tensor_size)
    output1, _, _, _, _ = model(input, 0)
    output2, _, _, _, _ = model(input, 1)
    output3, _, _, _, _ = model(input, 2)

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal,
        output1.detach(),
        output2.detach())

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal,
        output1.detach(),
        output3.detach())

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal,
        output2.detach(),
        output3.detach())


@pytest.mark.medium
def test_deterministic_ensemble_overfits_on_batch():
    n_networks = 5
    torch.manual_seed(0)

    model = EnvironmentModel(obs_dim=3, act_dim=4, n_networks=n_networks)

    x = torch.rand((10, 7))
    y = torch.rand((10, 5))
    lr = 1e-3

    optims = [Adam(model.networks[i_network].parameters(), lr=lr)
              for i_network in range(n_networks)]
    criterion = nn.MSELoss()

    y_pred = [0 for i in range(n_networks)]
    losses = [1e10 for i in range(n_networks)]

    for step in range(500):
        for i_network in range(n_networks):
            optim = optims[i_network]
            optim.zero_grad()
            y_pred, _, _, _, _ = model(x, i_network=i_network)
            loss = criterion(y_pred[:, :-1], y[:, :-1])
            loss.backward()
            optim.step()

            losses[i_network] = loss.item()

        print(losses)

    for i_network in range(n_networks):
        assert losses[i_network] < 2e-5


@pytest.mark.fast
def test_model_returns_prediction_of_random_network_if_not_specified():
    obs_dim = 5
    act_dim = 6

    torch.manual_seed(0)

    model = EnvironmentModel(obs_dim, act_dim, n_networks=40)

    tensor_size = (3, obs_dim+act_dim)
    input = torch.rand(tensor_size)
    output1 = model.get_prediction(input).detach().numpy()
    output2 = model.get_prediction(input).detach().numpy()

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal,
        output1,
        output2)


@pytest.mark.fast
def test_model_returns_same_output_if_network_specified():
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(obs_dim, act_dim, n_networks=10)

    tensor_size = (3, obs_dim+act_dim)
    input = torch.rand(tensor_size)
    output1 = model.get_prediction(input, i_network=5).detach().numpy()
    output2 = model.get_prediction(input, i_network=5).detach().numpy()

    np.testing.assert_array_equal(output1, output2)


@pytest.mark.fast
def test_deterministic_model_returns_binary_done_signal():
    obs_dim = 5
    act_dim = 6
    torch.manual_seed(2)

    model = EnvironmentModel(obs_dim, act_dim)

    tensor_size = (100, obs_dim+act_dim)
    input = torch.rand(tensor_size)
    output = model.get_prediction(input)

    for value in output[:, -1]:
        assert (value == 0 or value == 1)


@pytest.mark.fast
def test_probabilistic_model_returns_binary_done_signal():
    obs_dim = 5
    act_dim = 6
    torch.manual_seed(0)

    model = EnvironmentModel(obs_dim, act_dim, type='probabilistic')

    tensor_size = (100, obs_dim+act_dim)
    input = torch.rand(tensor_size)
    output = model.get_prediction(input).detach().numpy()

    assert output[:, -1].any()

    for value in output[:, -1]:
        assert (value == 0 or value == 1)


@pytest.mark.fast
def test_deterministic_model_returns_binary_done_signal_when_term_fn_used():
    obs_dim = 5
    act_dim = 6
    torch.manual_seed(2)

    model = EnvironmentModel(obs_dim, act_dim)

    tensor_size = (100, obs_dim+act_dim)
    input = torch.rand(tensor_size)
    output = model.get_prediction(input,
                                  term_fn=termination_functions['hopper'])

    for value in output[:, -1]:
        assert (value == 0 or value == 1)


@pytest.mark.medium
def test_deterministic_model_does_not_always_output_terminal():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    env = gym.make('hopper-random-v0')
    real_buffer, obs_dim, act_dim = load_dataset_from_env(
        env, n_samples=10000, buffer_device=device)
    model = EnvironmentModel(obs_dim, act_dim, type='deterministic')
    model.to(device)
    optim = Adam(model.parameters(), lr=1e-3)

    for i in range(500):

        x, y = get_x_y_from_batch(real_buffer.sample_train_batch(256, 0),
                                  device)

        optim.zero_grad()
        loss = deterministic_loss(x, y, model)

        if i % 100 == 0:
            print("Step: {} Loss: {:.3f}".format(i, loss.item()))
        loss.backward(retain_graph=True)
        optim.step()

    # Generate virtual rollouts and make sure that not everything is a terminal
    # state
    agent = RandomAgent(env, device=device)
    virtual_buffer = ReplayBuffer(obs_dim, act_dim, 10000, device=device)

    for model_rollout in range(10):
        rollout = generate_virtual_rollouts(
            model,
            agent,
            real_buffer,
            50,
            term_fn=termination_functions['hopper'])

        for i in range(len(rollout['obs'])):
            virtual_buffer.store(rollout['obs'][i],
                                 rollout['act'][i],
                                 rollout['rew'][i],
                                 rollout['next_obs'][i],
                                 rollout['done'][i],)

    terminal_ratio = virtual_buffer.get_terminal_ratio()

    print(terminal_ratio)
    assert terminal_ratio < 1
    assert terminal_ratio > 0


@pytest.mark.medium
def test_probabilistic_model_does_not_always_output_terminal():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    env = gym.make('hopper-random-v0')
    real_buffer, obs_dim, act_dim = load_dataset_from_env(
        env, 10000, buffer_device=device)
    model = EnvironmentModel(obs_dim, act_dim, type='probabilistic')
    model.to(device)
    optim = Adam(model.parameters(), lr=1e-3)

    for i in range(500):

        x, y = get_x_y_from_batch(real_buffer.sample_train_batch(256, 0),
                                  device)

        optim.zero_grad()
        loss = probabilistic_loss(x, y, model)

        if i % 100 == 0:
            print("Step: {} Loss: {:.3f}".format(i, loss.item()))
        loss.backward(retain_graph=True)
        optim.step()

    # Generate virtual rollouts and make sure that not everything is a terminal
    # state
    agent = RandomAgent(env, device=device)
    virtual_buffer = ReplayBuffer(obs_dim, act_dim, 10000, device=device)

    for model_rollout in range(10):

        rollout = generate_virtual_rollouts(
            model,
            agent,
            real_buffer,
            50,
            term_fn=termination_functions['hopper'])

        for i in range(len(rollout['obs'])):
            virtual_buffer.store(rollout['obs'][i],
                                 rollout['act'][i],
                                 rollout['rew'][i],
                                 rollout['next_obs'][i],
                                 rollout['done'][i],)

    terminal_ratio = virtual_buffer.get_terminal_ratio()

    print(terminal_ratio)
    assert terminal_ratio < 1
    assert terminal_ratio > 0


@pytest.mark.fast
def test_pessimistic_prediction_throws_error_if_model_not_probabilistic():
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(obs_dim, act_dim, [2, 2])

    tensor_size = (3, obs_dim+act_dim)
    input = torch.rand(tensor_size)

    with pytest.raises(ValueError):
        model.get_prediction(input, pessimism=1)


@pytest.mark.fast
def test_get_prediction_from_pessimistic_model():
    obs_dim = 5
    act_dim = 6
    n_samples = 100

    model = EnvironmentModel(obs_dim,
                             act_dim,
                             hidden=[2, 2],
                             type='probabilistic',
                             n_networks=2)

    tensor_size = (n_samples, obs_dim+act_dim)
    input = torch.rand(tensor_size)

    torch.random.manual_seed(0)

    optimistic_output1 = model.get_prediction(input,
                                              pessimism=0)

    torch.random.manual_seed(0)

    optimistic_output2 = model.get_prediction(input,
                                              pessimism=0)

    torch.random.manual_seed(0)

    pessimistic_output = model.get_prediction(input,
                                              pessimism=1)

    np.testing.assert_array_equal(optimistic_output1,
                                  optimistic_output2)

    np.testing.assert_array_equal(pessimistic_output.shape,
                                  optimistic_output1.shape)

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal,
        optimistic_output1,
        pessimistic_output)
