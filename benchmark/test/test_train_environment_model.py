from benchmark.utils.load_dataset import load_dataset_from_env
import gym
import torch
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.train_environment_model import train_environment_model
import pytest


def test_train_deterministic_environment_model():
    env = gym.make('halfcheetah-random-v0')
    buffer, obs_dim, act_dim = load_dataset_from_env(env)

    model = EnvironmentModel(obs_dim, act_dim)

    loss = train_environment_model(
        model, buffer, val_split=0.2, patience=2, debug=True)

    assert loss < 0.3


def test_raise_error_if_data_not_enough_for_split_at_given_batch_size():
    env = gym.make('halfcheetah-random-v0')
    buffer, obs_dim, act_dim = load_dataset_from_env(env, n_samples=100)

    with pytest.raises(ValueError):
        model = EnvironmentModel(obs_dim, act_dim)

        train_environment_model(
            model, buffer, val_split=0.2, patience=10, debug=True)


def test_train_probabilistic_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = gym.make('halfcheetah-random-v0')
    buffer, obs_dim, act_dim = load_dataset_from_env(env)

    model = EnvironmentModel(
        obs_dim, act_dim, type='probabilistic')

    model.to(device)

    loss = train_environment_model(
        model, buffer, val_split=0.2, patience=20, debug=True)

    assert loss < 0.8
