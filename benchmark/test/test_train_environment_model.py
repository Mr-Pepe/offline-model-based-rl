from benchmark.utils.load_dataset import load_dataset_from_env
import gym
import torch
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.train_environment_model import train_environment_model
import pytest


def test_train_deterministic_environment_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    env = gym.make('halfcheetah-random-v0')
    buffer, obs_dim, act_dim = load_dataset_from_env(env)

    model = EnvironmentModel(obs_dim, act_dim)
    model.to(device)

    val_losses = train_environment_model(
        model, buffer, val_split=0.2, patience=5, debug=True)

    for val_loss in val_losses:
        assert val_loss < 0.3


def test_raise_error_if_data_not_enough_for_split_at_given_batch_size():
    env = gym.make('halfcheetah-random-v0')
    buffer, obs_dim, act_dim = load_dataset_from_env(env, n_samples=100)

    with pytest.raises(ValueError):
        model = EnvironmentModel(obs_dim, act_dim)

        train_environment_model(model,
                                buffer,
                                val_split=0.2,
                                patience=10,
                                debug=True)


def test_train_probabilistic_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    env = gym.make('halfcheetah-random-v0')
    buffer, obs_dim, act_dim = load_dataset_from_env(env)

    model = EnvironmentModel(obs_dim, act_dim, type='probabilistic')

    model.to(device)

    val_losses = train_environment_model(model,
                                         buffer,
                                         val_split=0.2,
                                         patience=20,
                                         debug=True)

    for val_loss in val_losses:
        assert val_loss < 0.8


def test_train_deterministic_ensemble():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    env = gym.make('halfcheetah-random-v0')
    buffer, obs_dim, act_dim = load_dataset_from_env(env)

    model = EnvironmentModel(obs_dim, act_dim, n_networks=5)
    model.to(device)

    val_losses = train_environment_model(model,
                                         buffer,
                                         val_split=0.2,
                                         patience=5,
                                         debug=True)

    for val_loss in val_losses:
        assert val_loss < 0.3

    assert len(set(val_losses)) == len(val_losses)


def test_train_probabilistic_ensemble():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    env = gym.make('halfcheetah-random-v0')
    buffer, obs_dim, act_dim = load_dataset_from_env(env)

    model = EnvironmentModel(obs_dim,
                             act_dim,
                             type='probabilistic',
                             n_networks=2)

    model.to(device)

    val_losses = train_environment_model(model,
                                         buffer,
                                         val_split=0.2,
                                         patience=20,
                                         debug=True)

    for val_loss in val_losses:
        assert val_loss < 0.8

    assert len(set(val_losses)) == len(val_losses)
