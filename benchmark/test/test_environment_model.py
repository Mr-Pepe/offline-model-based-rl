from benchmark.utils.envs import HALF_CHEETAH_RANDOM, HOPPER_RANDOM
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
from benchmark.utils.postprocessing import postprocessing_functions
import torch.nn as nn
import torch
from torch.optim.adam import Adam
import d4rl  # noqa
import gym
import numpy as np
from math import pi as PI
from matplotlib.pyplot import cm

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
    output = model.get_prediction(input)

    np.testing.assert_array_equal(output.shape, (3, obs_dim+2))


@pytest.mark.medium
def test_single_deterministic_network_overfits_on_single_sample():
    torch.manual_seed(0)
    model = EnvironmentModel(1, 1)

    x = torch.as_tensor([3, 3], dtype=torch.float32)
    y = torch.as_tensor([5, 4], dtype=torch.float32)
    lr = 1e-2

    optim = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    y_pred = 0

    for i in range(1000):
        optim.zero_grad()
        y_pred, _, _, _, _ = model(torch.stack(3 * [x]))
        loss = criterion(y_pred, torch.stack(3 * [y]).unsqueeze(0))
        print("Loss: {}".format(loss))
        loss.backward()
        optim.step()

    assert criterion(y_pred, torch.stack(3 * [y]).unsqueeze(0)).item() < 1e-3


@pytest.mark.medium
def test_single_deterministic_network_overfits_on_batch():
    model = EnvironmentModel(obs_dim=3, act_dim=4)

    x = torch.rand((10, 7))
    y = torch.rand((10, 4))
    lr = 1e-2

    optim = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    y_pred = 0

    for i in range(1000):
        optim.zero_grad()
        y_pred, _, _, _, _ = model(x)
        loss = criterion(y_pred, y.unsqueeze(0))
        print("Loss: {}".format(loss))
        loss.backward()
        optim.step()

    assert criterion(y_pred, y.unsqueeze(0)).item() < 1e-5


@pytest.mark.medium
def test_deterministic_model_trains_on_offline_data():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = gym.make(HALF_CHEETAH_RANDOM)
    dataset = env.get_dataset()
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards'].reshape(-1, 1)

    model = EnvironmentModel(observations.shape[1], actions.shape[1])

    model.to(device)

    lr = 1e-2
    batch_size = 1024

    obs_rew_optim = Adam(model.parameters(), lr=lr)
    obs_rew_criterion = nn.MSELoss()

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

        obs_rew_optim.zero_grad()
        y_pred, _, _, _, _ = model(x)
        loss = obs_rew_criterion(y_pred, y.unsqueeze(0))
        print("Obs/Reward loss: {}".format(loss))
        losses.append(loss.item())
        loss.backward()
        obs_rew_optim.step()

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


@pytest.mark.slow
def test_probabilistic_model_trains_on_toy_dataset(steps=3000, plot=False, augment_loss=False,
                                                   bounds_trainable=True, steps_per_plot=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(0)

    x = torch.rand((1000,)) * PI - 2*PI
    x = torch.cat((x, torch.rand((1000,)) * PI + PI)).to(device)
    y = torch.sin(x) + torch.normal(0, 0.225 *
                                    torch.abs(torch.sin(1.5*x + PI/8)))\
        .to(device)

    n_networks = 4

    buffer = ReplayBuffer(1, 1, size=100000, device=device)
    buffer.obs_buf = x.unsqueeze(-1)
    buffer.obs2_buf = y.unsqueeze(-1)
    buffer.rew_buf = y
    buffer.size = x.numel()

    model = EnvironmentModel(
        1, 1, hidden=[200, 200, 200, 200], type='probabilistic',
        n_networks=n_networks,
        device=device,
        obs_bounds_trainable=bounds_trainable,
        r_bounds_trainable=bounds_trainable)

    x_true = torch.arange(-3*PI, 3*PI, 0.01)
    y_true = torch.sin(x_true)
    f, ax = plt.subplots(1, 1)

    for i in range(steps):
        model.train_to_convergence(
            buffer, lr=1e-4, debug=True, max_n_train_batches=steps_per_plot, batch_size=10,
            augment_loss=augment_loss)

        if plot:
            _, mean_plt, logvar_plt, max_logvar_plt, _ = model(
                torch.cat((x_true.unsqueeze(-1), torch.zeros_like(x_true.unsqueeze(-1))), dim=1))
            mean_plt = mean_plt[:, :, 1].detach().cpu()
            logvar_plt = logvar_plt[:, :, 1].detach().cpu()
            max_std = torch.exp(0.5*max_logvar_plt[:, 1].detach().cpu())

            print(max_logvar_plt)

            std = torch.exp(0.5*logvar_plt)

            x_plt = x.cpu()
            y_plt = y.cpu()

            # ax = axs[0]
            ax.clear()
            ax.scatter(x_plt[800:1200], y_plt[800:1200],
                       color='green', marker='x', s=5)
            ax.plot(x_true, y_true, color='black')

            color = cm.rainbow(np.linspace(0, 1, n_networks))
            for i_network, c in zip(range(n_networks),color):

                # ax.fill_between(x_true, (mean_plt[i_network]+max_std[i_network]).view(-1),
                #                 (mean_plt[i_network]-max_std[i_network]).view(-1),
                #                 color='grey', zorder=-2)
                ax.fill_between(x_true, (mean_plt[i_network]+std[i_network]).view(-1), (mean_plt[i_network]-std[i_network]).view(-1),
                                color=c, alpha=0.2)
                ax.plot(x_true, mean_plt[i_network].view(-1), color=c)
                ax.set_ylim([-3, 3])

            plt.draw()
            plt.pause(0.001)


@pytest.mark.fast
def test_deterministic_ensemble_gives_different_predictions_per_model():
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(obs_dim, act_dim, n_networks=3)

    tensor_size = (3, obs_dim+act_dim)
    input = torch.rand(tensor_size)
    output1 = model.get_prediction(input, 0)
    output2 = model.get_prediction(input, 1)
    output3 = model.get_prediction(input, 2)

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
    y = torch.rand((10, 4))
    lr = 1e-2

    optim = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss = torch.as_tensor(0)

    for step in range(500):
        optim.zero_grad()
        y_pred, _, _, _, _ = model(x)
        loss = criterion(y_pred, torch.stack(n_networks*[y]))
        loss.backward()
        optim.step()

        print(loss.item())

    assert loss.item() < 1e-5


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
    output = model.get_prediction(input, 0)

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
    output = model.get_prediction(input, 0).detach().numpy()

    for value in output[:, -1]:
        assert (value == 0 or value == 1)


@pytest.mark.fast
def test_deterministic_model_returns_binary_done_signal_when_post_fn_used():
    obs_dim = 5
    act_dim = 6
    torch.manual_seed(2)

    model = EnvironmentModel(obs_dim, act_dim,
                             post_fn=postprocessing_functions['hopper'])

    tensor_size = (100, obs_dim+act_dim)
    input = torch.rand(tensor_size)
    output = model.get_prediction(input, 0)

    for value in output[:, -1]:
        assert (value == 0 or value == 1)


@pytest.mark.medium
def test_deterministic_model_does_not_always_output_terminal():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    env = gym.make(HOPPER_RANDOM)
    real_buffer, obs_dim, act_dim = load_dataset_from_env(
        env, n_samples=10000, buffer_device=device)
    model = EnvironmentModel(obs_dim, act_dim, type='deterministic',
                             post_fn=postprocessing_functions['hopper'],
                             device=device)
    optim = Adam(model.parameters(), lr=1e-2)

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
        rollout, _ = generate_virtual_rollouts(
            model,
            agent,
            real_buffer,
            50
        )

        for i in range(len(rollout['obs'])):
            virtual_buffer.store(rollout['obs'][i],
                                 rollout['act'][i],
                                 rollout['rew'][i],
                                 rollout['next_obs'][i],
                                 rollout['done'][i],)

    terminal_ratio = virtual_buffer.get_terminal_ratio()

    print("Terminal ratio: {}".format(terminal_ratio))
    assert terminal_ratio < 1
    assert terminal_ratio > 0


@pytest.mark.medium
def test_probabilistic_model_does_not_always_output_terminal():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    env = gym.make(HOPPER_RANDOM)
    real_buffer, obs_dim, act_dim = load_dataset_from_env(
        env, 10000, buffer_device=device)
    model = EnvironmentModel(obs_dim, act_dim, type='probabilistic',
                             post_fn=postprocessing_functions['hopper'],
                             device=device)
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

        rollout, _ = generate_virtual_rollouts(
            model,
            agent,
            real_buffer,
            50
        )

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
def test_aleatoric_pessimism_throws_error_if_model_not_probabilistic():
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(obs_dim, act_dim, [2, 2])

    tensor_size = (3, obs_dim+act_dim)
    input = torch.rand(tensor_size)

    with pytest.raises(ValueError):
        model.get_prediction(input, pessimism=1, mode='mopo')


@pytest.mark.fast
def test_throws_error_if_mode_unknown():
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(obs_dim, act_dim, [2, 2])

    tensor_size = (3, obs_dim+act_dim)
    input = torch.rand(tensor_size)

    with pytest.raises(ValueError):
        model.get_prediction(input, pessimism=1, mode='epistemi')


@pytest.mark.fast
def test_get_prediction_from_pessimistic_model():
    obs_dim = 5
    act_dim = 6
    n_samples = 100
    n_networks = 2

    model = EnvironmentModel(obs_dim,
                             act_dim,
                             hidden=[2, 2],
                             type='probabilistic',
                             n_networks=n_networks)

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
