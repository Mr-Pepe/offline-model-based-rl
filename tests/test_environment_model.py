# pylint: disable=unused-import
import gym
import numpy as np
import pytest
import torch
from d4rl.offline_env import OfflineEnv
from torch import nn
from torch.optim.adam import Adam

from offline_mbrl.actors.random_agent import RandomAgent
from offline_mbrl.models.environment_model import (
    EnvironmentModel,
    get_model_input_and_ground_truth_from_batch,
)
from offline_mbrl.schemas import EnvironmentModelConfiguration
from offline_mbrl.utils.envs import HOPPER_MEDIUM_REPLAY_V2
from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.modes import ALEATORIC_PENALTY
from offline_mbrl.utils.replay_buffer import ReplayBuffer
from offline_mbrl.utils.termination_functions import termination_functions
from offline_mbrl.utils.virtual_rollouts import generate_virtual_rollouts

gym.logger.set_level(40)


@pytest.mark.fast
def test_is_nn_module() -> None:
    assert issubclass(EnvironmentModel, nn.Module)


@pytest.mark.fast
def test_takes_state_and_action_as_input_and_outputs_state_reward_done() -> None:
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(
        obs_dim, act_dim, EnvironmentModelConfiguration(hidden_layer_sizes=(2, 2))
    )

    tensor_size = (3, obs_dim + act_dim)
    obs_act = torch.rand(tensor_size)
    output = model.get_prediction(obs_act)

    assert isinstance(output, torch.Tensor)

    np.testing.assert_array_equal(output.shape, (3, obs_dim + 2))


@pytest.mark.medium
def test_single_deterministic_network_overfits_on_single_sample() -> None:
    torch.manual_seed(0)
    model = EnvironmentModel(1, 1)

    x = torch.as_tensor([3, 3], dtype=torch.float32)
    y = torch.as_tensor([5, 4], dtype=torch.float32)
    lr = 1e-2

    optim = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    y_pred = 0

    for _ in range(1000):
        optim.zero_grad()
        y_pred, _, _, _, _ = model(torch.stack(3 * [x]))
        loss = criterion(y_pred, torch.stack(3 * [y]).unsqueeze(0))
        print(f"Loss: {loss}")
        loss.backward()
        optim.step()

    assert criterion(y_pred, torch.stack(3 * [y]).unsqueeze(0)).item() < 1e-3


@pytest.mark.medium
def test_single_deterministic_network_overfits_on_batch() -> None:
    model = EnvironmentModel(obs_dim=3, act_dim=4)

    x = torch.rand((10, 7))
    y = torch.rand((10, 4))
    lr = 1e-2

    optim = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    y_pred = 0

    for _ in range(1000):
        optim.zero_grad()
        y_pred, _, _, _, _ = model(x)
        loss = criterion(y_pred, y.unsqueeze(0))
        print(f"Loss: {loss}")
        loss.backward()
        optim.step()

    assert criterion(y_pred, y.unsqueeze(0)).item() < 1e-4


@pytest.mark.medium
def test_deterministic_model_trains_on_offline_data() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env: OfflineEnv = gym.make(HOPPER_MEDIUM_REPLAY_V2)
    dataset = env.get_dataset()
    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"].reshape(-1, 1)

    model = EnvironmentModel(observations.shape[1], actions.shape[1])

    model.to(device)

    lr = 1e-2
    batch_size = 1024

    obs_rew_optim = Adam(model.parameters(), lr=lr)
    obs_rew_criterion = nn.MSELoss()

    losses = []

    for _ in range(2000):
        idxs = np.random.randint(0, observations.shape[0] - 1, size=batch_size)
        x = torch.as_tensor(
            np.concatenate((observations[idxs], actions[idxs]), axis=1),
            dtype=torch.float32,
            device=torch.device(device),
        )
        y = torch.as_tensor(
            np.concatenate((observations[idxs + 1], rewards[idxs]), axis=1),
            dtype=torch.float32,
            device=torch.device(device),
        )

        obs_rew_optim.zero_grad()
        y_pred, _, _, _, _ = model(x)
        loss = obs_rew_criterion(y_pred, y.unsqueeze(0))
        print(f"Obs/Reward loss: {loss}")
        losses.append(loss.item())
        loss.backward()
        obs_rew_optim.step()

    assert losses[-1] < 1


@pytest.mark.fast
def test_probabilistic_model_returns_different_results_for_same_input() -> None:
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(
        obs_dim,
        act_dim,
        EnvironmentModelConfiguration(type="probabilistic", hidden_layer_sizes=(2, 2)),
    )

    tensor_size = (3, obs_dim + act_dim)
    obs_act = torch.rand(tensor_size)
    output1, _, _, _, _ = model(obs_act)
    output2, _, _, _, _ = model(obs_act)

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        output1.detach(),
        output2.detach(),
    )


@pytest.mark.fast
def test_raises_error_if_type_unknown() -> None:
    with pytest.raises(ValueError):
        EnvironmentModel(1, 2, EnvironmentModelConfiguration(type="asdasd"))


@pytest.mark.fast
def test_deterministic_ensemble_gives_different_predictions_per_model() -> None:
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(
        obs_dim, act_dim, EnvironmentModelConfiguration(n_networks=3)
    )

    tensor_size = (3, obs_dim + act_dim)
    obs_act = torch.rand(tensor_size)
    output1 = model.get_prediction(obs_act, 0)
    output2 = model.get_prediction(obs_act, 1)
    output3 = model.get_prediction(obs_act, 2)

    assert isinstance(output1, torch.Tensor)
    assert isinstance(output2, torch.Tensor)
    assert isinstance(output3, torch.Tensor)

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        output1.detach(),
        output2.detach(),
    )

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        output1.detach(),
        output3.detach(),
    )

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        output2.detach(),
        output3.detach(),
    )


@pytest.mark.medium
def test_deterministic_ensemble_overfits_on_batch() -> None:
    n_networks = 5
    torch.manual_seed(0)

    model = EnvironmentModel(3, 4, EnvironmentModelConfiguration(n_networks=n_networks))

    x = torch.rand((10, 7))
    y = torch.rand((10, 4))
    lr = 1e-2

    optim = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss = torch.as_tensor(0)

    for _ in range(500):
        optim.zero_grad()
        y_pred, _, _, _, _ = model(x)
        loss = criterion(y_pred, torch.stack(n_networks * [y]))
        loss.backward()
        optim.step()

        print(loss.item())

    assert loss.item() < 3e-4


@pytest.mark.fast
def test_model_returns_prediction_of_random_network_if_not_specified() -> None:
    obs_dim = 5
    act_dim = 6

    torch.manual_seed(0)

    model = EnvironmentModel(
        obs_dim, act_dim, EnvironmentModelConfiguration(n_networks=40)
    )

    tensor_size = (3, obs_dim + act_dim)
    obs_act = torch.rand(tensor_size)
    output1 = model.get_prediction(obs_act).detach().numpy()  # type: ignore
    output2 = model.get_prediction(obs_act).detach().numpy()  # type: ignore

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, output1, output2
    )


@pytest.mark.fast
def test_model_returns_same_output_if_network_specified() -> None:
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(
        obs_dim, act_dim, EnvironmentModelConfiguration(n_networks=10)
    )

    tensor_size = (3, obs_dim + act_dim)
    obs_act = torch.rand(tensor_size)
    output1 = model.get_prediction(obs_act, 5).detach().numpy()  # type: ignore
    output2 = model.get_prediction(obs_act, 5).detach().numpy()  # type: ignore

    np.testing.assert_array_equal(output1, output2)


@pytest.mark.fast
def test_deterministic_model_returns_binary_done_signal() -> None:
    obs_dim = 5
    act_dim = 6
    torch.manual_seed(2)

    model = EnvironmentModel(obs_dim, act_dim)

    tensor_size = (100, obs_dim + act_dim)
    obs_act = torch.rand(tensor_size)
    output = model.get_prediction(obs_act, 0)

    assert isinstance(output, torch.Tensor)

    for value in output[:, -1]:
        assert value in (0, 1)


@pytest.mark.fast
def test_probabilistic_model_returns_binary_done_signal() -> None:
    obs_dim = 5
    act_dim = 6
    torch.manual_seed(0)

    model = EnvironmentModel(
        obs_dim, act_dim, EnvironmentModelConfiguration(type="probabilistic")
    )

    tensor_size = (100, obs_dim + act_dim)
    obs_act = torch.rand(tensor_size)
    output = model.get_prediction(obs_act, 0).detach().numpy()  # type: ignore

    assert isinstance(output, np.ndarray)

    for value in output[:, -1]:
        assert value in (0, 1)


@pytest.mark.fast
def test_deterministic_model_returns_binary_done_signal_when_using_term_fn() -> None:
    obs_dim = 5
    act_dim = 6
    torch.manual_seed(2)

    model = EnvironmentModel(
        obs_dim,
        act_dim,
        EnvironmentModelConfiguration(
            termination_function=termination_functions["hopper"]
        ),
    )

    tensor_size = (100, obs_dim + act_dim)
    obs_act = torch.rand(tensor_size)
    output = model.get_prediction(obs_act, 0)

    assert isinstance(output, torch.Tensor)

    for value in output[:, -1]:
        assert value in (0, 1)


@pytest.mark.medium
def test_deterministic_model_does_not_always_output_terminal() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = HOPPER_MEDIUM_REPLAY_V2
    torch.manual_seed(0)

    env = gym.make(HOPPER_MEDIUM_REPLAY_V2)
    real_buffer, obs_dim, act_dim = load_dataset_from_env(
        env_name, n_samples=10000, buffer_device=device
    )
    model = EnvironmentModel(
        obs_dim,
        act_dim,
        EnvironmentModelConfiguration(
            type="deterministic",
            termination_function=termination_functions["hopper"],
            device=device,
        ),
    )
    optim = Adam(model.parameters(), lr=1e-2)

    for step in range(500):

        x, y = get_model_input_and_ground_truth_from_batch(
            real_buffer.sample_train_batch(256, 0), device
        )

        optim.zero_grad()
        loss = model.deterministic_loss(x, y)

        if step % 100 == 0:
            print(f"Step: {step} Loss: {loss.item():.3f}")
        loss.backward(retain_graph=True)
        optim.step()

    # Generate virtual rollouts and make sure that not everything is a terminal
    # state
    agent = RandomAgent(env, device=device)
    virtual_buffer = ReplayBuffer(obs_dim, act_dim, 10000, device=device)

    for _ in range(10):
        rollout, _ = generate_virtual_rollouts(model, agent, real_buffer, 50)

        for step in range(len(rollout["obs"])):
            virtual_buffer.store(
                rollout["obs"][step],
                rollout["act"][step],
                rollout["rew"][step],
                rollout["next_obs"][step],
                rollout["done"][step],
            )

    terminal_ratio = virtual_buffer.get_terminal_ratio()

    print(f"Terminal ratio: {terminal_ratio}")
    assert terminal_ratio < 1
    assert terminal_ratio > 0


@pytest.mark.medium
def test_probabilistic_model_does_not_always_output_terminal() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = HOPPER_MEDIUM_REPLAY_V2
    torch.manual_seed(0)

    env = gym.make(env_name)
    real_buffer, obs_dim, act_dim = load_dataset_from_env(
        env_name, 10000, buffer_device=device
    )
    model = EnvironmentModel(
        obs_dim,
        act_dim,
        EnvironmentModelConfiguration(
            type="probabilistic",
            termination_function=termination_functions["hopper"],
            device=device,
        ),
    )
    optim = Adam(model.parameters(), lr=1e-3)

    for step in range(500):

        x, y = get_model_input_and_ground_truth_from_batch(
            real_buffer.sample_train_batch(256, 0), device
        )

        optim.zero_grad()
        loss = model.probabilistic_loss(x, y)

        if step % 100 == 0:
            print(f"Step: {step} Loss: {loss.item():.3f}")
        loss.backward(retain_graph=True)
        optim.step()

    # Generate virtual rollouts and make sure that not everything is a terminal
    # state
    agent = RandomAgent(env, device=device)
    virtual_buffer = ReplayBuffer(obs_dim, act_dim, 10000, device=device)

    for _ in range(10):
        rollout, _ = generate_virtual_rollouts(model, agent, real_buffer, 50)

        for step in range(len(rollout["obs"])):
            virtual_buffer.store(
                rollout["obs"][step],
                rollout["act"][step],
                rollout["rew"][step],
                rollout["next_obs"][step],
                rollout["done"][step],
            )

    terminal_ratio = virtual_buffer.get_terminal_ratio()

    print(terminal_ratio)
    assert terminal_ratio < 1
    assert terminal_ratio > 0


@pytest.mark.fast
def test_aleatoric_pessimism_throws_error_if_model_not_probabilistic() -> None:
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(
        obs_dim, act_dim, EnvironmentModelConfiguration(hidden_layer_sizes=(2, 2))
    )

    tensor_size = (3, obs_dim + act_dim)
    obs_act = torch.rand(tensor_size)

    with pytest.raises(ValueError):
        model.get_prediction(obs_act, pessimism=1, mode=ALEATORIC_PENALTY)


@pytest.mark.fast
def test_throws_error_if_mode_unknown() -> None:
    obs_dim = 5
    act_dim = 6

    model = EnvironmentModel(
        obs_dim, act_dim, EnvironmentModelConfiguration(hidden_layer_sizes=(2, 2))
    )

    tensor_size = (3, obs_dim + act_dim)
    obs_act = torch.rand(tensor_size)

    with pytest.raises(ValueError):
        model.get_prediction(obs_act, pessimism=1, mode="epistemi")


@pytest.mark.fast
def test_get_prediction_from_pessimistic_model() -> None:
    obs_dim = 5
    act_dim = 6
    n_samples = 100
    n_networks = 2

    model = EnvironmentModel(
        obs_dim,
        act_dim,
        EnvironmentModelConfiguration(
            hidden_layer_sizes=[2, 2],
            type="probabilistic",
            n_networks=n_networks,
        ),
    )

    tensor_size = (n_samples, obs_dim + act_dim)
    obs_act = torch.rand(tensor_size)

    torch.random.manual_seed(0)

    optimistic_output1 = model.get_prediction(obs_act, pessimism=0)

    torch.random.manual_seed(0)

    optimistic_output2 = model.get_prediction(obs_act, pessimism=0)

    torch.random.manual_seed(0)

    pessimistic_output = model.get_prediction(
        obs_act, pessimism=1, mode=ALEATORIC_PENALTY
    )

    np.testing.assert_array_equal(optimistic_output1, optimistic_output2)

    assert isinstance(pessimistic_output, torch.Tensor)
    assert isinstance(optimistic_output1, torch.Tensor)

    np.testing.assert_array_equal(pessimistic_output.shape, optimistic_output1.shape)

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        optimistic_output1,
        pessimistic_output,
    )
