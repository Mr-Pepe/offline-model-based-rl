import gym
import pytest
import torch

from offline_mbrl.models.environment_model import EnvironmentModel
from offline_mbrl.utils.envs import HALF_CHEETAH_RANDOM_V2
from offline_mbrl.utils.load_dataset import load_dataset_from_env


@pytest.mark.slow
def test_train_deterministic_environment_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    env = gym.make(HALF_CHEETAH_RANDOM_V2)
    buffer, obs_dim, act_dim = load_dataset_from_env(
        env, buffer_device=device, n_samples=100000
    )

    model = EnvironmentModel(obs_dim, act_dim)
    model.to(device)

    val_losses, _ = model.train_to_convergence(
        buffer, val_split=0.2, patience=3, debug=True
    )

    for val_loss in val_losses:
        assert val_loss < 0.6


@pytest.mark.slow
def test_train_probabilistic_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    env = gym.make(HALF_CHEETAH_RANDOM_V2)
    buffer, obs_dim, act_dim = load_dataset_from_env(
        env, buffer_device=device, n_samples=100000
    )

    model = EnvironmentModel(obs_dim, act_dim, model_type="probabilistic")

    model.to(device)

    val_losses, _ = model.train_to_convergence(
        buffer, val_split=0.2, patience=10, debug=True
    )

    for val_loss in val_losses:
        assert val_loss < 0.6


@pytest.mark.slow
def test_train_deterministic_ensemble():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    env = gym.make(HALF_CHEETAH_RANDOM_V2)
    buffer, obs_dim, act_dim = load_dataset_from_env(
        env, buffer_device=device, n_samples=100000
    )

    model = EnvironmentModel(obs_dim, act_dim, n_networks=2)
    model.to(device)

    val_losses, _ = model.train_to_convergence(
        buffer, val_split=0.2, patience=5, debug=True
    )

    for val_loss in val_losses:
        assert val_loss < 0.6

    assert len(set(val_losses)) == len(val_losses)


@pytest.mark.slow
def test_train_probabilistic_ensemble():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    env = gym.make(HALF_CHEETAH_RANDOM_V2)
    buffer, obs_dim, act_dim = load_dataset_from_env(
        env, buffer_device=device, n_samples=100000
    )

    model = EnvironmentModel(obs_dim, act_dim, model_type="probabilistic", n_networks=2)

    model.to(device)

    val_losses, _ = model.train_to_convergence(
        buffer, val_split=0.2, patience=10, debug=True
    )

    for val_loss in val_losses:
        assert val_loss < 0.6

    assert len(set(val_losses)) == len(val_losses)


@pytest.mark.medium
def test_training_stops_after_specified_number_of_batches():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = HALF_CHEETAH_RANDOM_V2
    torch.manual_seed(0)

    env = gym.make(env_name)
    buffer, obs_dim, act_dim = load_dataset_from_env(
        env_name, buffer_device=device, n_samples=100000
    )

    model = EnvironmentModel(obs_dim, act_dim, n_networks=5)
    model.to(device)

    _, n_train_batches = model.train_to_convergence(
        buffer, val_split=0.2, patience=1, debug=True, max_n_train_batches=50
    )

    assert n_train_batches == 50
