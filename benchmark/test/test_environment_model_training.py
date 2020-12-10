from benchmark.utils.load_dataset import load_dataset_from_env
import gym
import torch
from benchmark.models.environment_model import EnvironmentModel
import pytest


@pytest.mark.slow
def test_train_deterministic_environment_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    env = gym.make('halfcheetah-random-v0')
    buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                     buffer_device=device,
                                                     n_samples=100000)

    model = EnvironmentModel(obs_dim, act_dim)
    model.to(device)

    val_losses, _ = model.train_to_convergence(buffer,
                                               val_split=0.2,
                                               patience=5,
                                               debug=True)

    for val_loss in val_losses:
        assert val_loss < 0.6


@pytest.mark.slow
def test_train_probabilistic_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    env = gym.make('halfcheetah-random-v0')
    buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                     buffer_device=device,
                                                     n_samples=100000)

    model = EnvironmentModel(obs_dim, act_dim, type='probabilistic')

    model.to(device)

    val_losses, _ = model.train_to_convergence(buffer,
                                               val_split=0.2,
                                               patience=10,
                                               debug=True)

    for val_loss in val_losses:
        assert val_loss < 0.6


@pytest.mark.slow
def test_train_deterministic_ensemble():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    env = gym.make('halfcheetah-random-v0')
    buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                     buffer_device=device,
                                                     n_samples=100000)

    model = EnvironmentModel(obs_dim, act_dim, n_networks=2)
    model.to(device)

    val_losses, _ = model.train_to_convergence(buffer,
                                               val_split=0.2,
                                               patience=5,
                                               debug=True)

    for val_loss in val_losses:
        assert val_loss < 0.6

    assert len(set(val_losses)) == len(val_losses)


@pytest.mark.slow
def test_train_probabilistic_ensemble():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    env = gym.make('halfcheetah-random-v0')
    buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                     buffer_device=device,
                                                     n_samples=100000)

    model = EnvironmentModel(obs_dim,
                             act_dim,
                             type='probabilistic',
                             n_networks=2)

    model.to(device)

    val_losses, _ = model.train_to_convergence(buffer,
                                               val_split=0.2,
                                               patience=10,
                                               debug=True)

    for val_loss in val_losses:
        assert val_loss < 0.6

    assert len(set(val_losses)) == len(val_losses)


@pytest.mark.slow
def test_patience_can_be_list():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    env = gym.make('halfcheetah-random-v0')
    buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                     buffer_device=device,
                                                     n_samples=100000)

    models = [EnvironmentModel(obs_dim, act_dim),
              EnvironmentModel(obs_dim, act_dim)]

    val_losses = []

    for i, model in enumerate(models):
        model.to(device)

        val_loss, _ = model.train_to_convergence(buffer,
                                                 val_split=0.2,
                                                 patience=[1, 3],
                                                 patience_value=i,
                                                 debug=True)

        val_losses.append(val_loss[0])

    assert val_losses[1] < val_losses[0]


@pytest.mark.medium
def test_training_stops_after_specified_number_of_batches():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    env = gym.make('halfcheetah-random-v0')
    buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                     buffer_device=device,
                                                     n_samples=100000)

    model = EnvironmentModel(obs_dim, act_dim, n_networks=5)
    model.to(device)

    val_losses, n_train_batches = model.train_to_convergence(
        buffer,
        val_split=0.2,
        patience=1,
        debug=True,
        max_n_train_batches=50
    )

    assert n_train_batches == 50
