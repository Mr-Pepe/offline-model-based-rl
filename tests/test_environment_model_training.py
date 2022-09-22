import pytest
import torch

from offline_mbrl.models.environment_model import EnvironmentModel
from offline_mbrl.schemas import EnvironmentModelConfiguration
from offline_mbrl.utils.envs import HALF_CHEETAH_MEDIUM_REPLAY_V2
from offline_mbrl.utils.load_dataset import load_dataset_from_env


@pytest.mark.slow
def test_train_deterministic_environment_model() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    buffer, obs_dim, act_dim = load_dataset_from_env(
        HALF_CHEETAH_MEDIUM_REPLAY_V2, buffer_device=device, n_samples=100000
    )

    model_config = EnvironmentModelConfiguration(val_split=0.2, training_patience=3)

    model = EnvironmentModel(obs_dim, act_dim, model_config)
    model.to(device)

    val_losses, _ = model.train_to_convergence(buffer, model_config, debug=True)

    for val_loss in val_losses:
        assert val_loss < 0.6


@pytest.mark.slow
def test_train_probabilistic_model() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    buffer, obs_dim, act_dim = load_dataset_from_env(
        HALF_CHEETAH_MEDIUM_REPLAY_V2, buffer_device=device, n_samples=100000
    )

    model_config = EnvironmentModelConfiguration(
        type="probabilistic",
        val_split=0.2,
        training_patience=10,
    )

    model = EnvironmentModel(obs_dim, act_dim, model_config)

    model.to(device)

    val_losses, _ = model.train_to_convergence(buffer, model_config, debug=True)

    for val_loss in val_losses:
        assert val_loss < 0.6


@pytest.mark.slow
def test_train_deterministic_ensemble() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    buffer, obs_dim, act_dim = load_dataset_from_env(
        HALF_CHEETAH_MEDIUM_REPLAY_V2, buffer_device=device, n_samples=100000
    )

    model_config = EnvironmentModelConfiguration(
        n_networks=2,
        val_split=0.2,
        training_patience=5,
    )

    model = EnvironmentModel(obs_dim, act_dim, model_config)
    model.to(device)

    val_losses, _ = model.train_to_convergence(buffer, model_config, debug=True)

    for val_loss in val_losses:
        assert val_loss < 0.6

    assert len(set(val_losses)) == len(val_losses)


@pytest.mark.slow
def test_train_probabilistic_ensemble() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    buffer, obs_dim, act_dim = load_dataset_from_env(
        HALF_CHEETAH_MEDIUM_REPLAY_V2, buffer_device=device, n_samples=100000
    )

    model_config = EnvironmentModelConfiguration(
        type="probabilistic",
        n_networks=2,
        val_split=0.2,
        training_patience=10,
    )

    model = EnvironmentModel(obs_dim, act_dim, model_config)

    model.to(device)

    val_losses, _ = model.train_to_convergence(buffer, model_config, debug=True)

    for val_loss in val_losses:
        assert val_loss < 0.6

    assert len(set(val_losses)) == len(val_losses)


@pytest.mark.medium
def test_training_stops_after_specified_number_of_batches() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = HALF_CHEETAH_MEDIUM_REPLAY_V2
    torch.manual_seed(0)

    buffer, obs_dim, act_dim = load_dataset_from_env(
        env_name, buffer_device=device, n_samples=100000
    )

    model_config = EnvironmentModelConfiguration(
        n_networks=5,
        val_split=0.2,
        training_patience=1,
        max_number_of_training_batches=50,
    )

    model = EnvironmentModel(obs_dim, act_dim, model_config)
    model.to(device)

    _, n_train_batches = model.train_to_convergence(
        buffer,
        model_config,
        debug=True,
    )

    assert n_train_batches == 50
