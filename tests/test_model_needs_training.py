import pytest

from offline_mbrl.models.environment_model import EnvironmentModel
from offline_mbrl.utils.model_needs_training import model_needs_training


@pytest.mark.fast
def test_returns_false_if_model_is_None() -> None:
    assert (
        model_needs_training(
            model=None,
            step=1,
            buffer_size=1,
            init_steps=0,
            steps_since_model_training=5,
            train_model_every=1,
        )
        is False
    )


@pytest.mark.fast
def test_returns_true_if_schdule_is_due() -> None:
    model = EnvironmentModel(1, 1)

    assert (
        model_needs_training(
            model,
            step=1,
            buffer_size=1,
            init_steps=0,
            steps_since_model_training=5,
            train_model_every=1,
        )
        is True
    )


@pytest.mark.fast
def test_returns_false_if_schdule_is_not_due() -> None:
    model = EnvironmentModel(1, 1)

    assert (
        model_needs_training(
            model,
            step=1,
            buffer_size=1,
            init_steps=0,
            steps_since_model_training=1,
            train_model_every=5,
        )
        is False
    )


@pytest.mark.fast
def test_returns_true_if_performing_offline_training() -> None:
    model = EnvironmentModel(1, 1)

    assert (
        model_needs_training(
            model,
            step=-1,
            buffer_size=1,
            init_steps=0,
            steps_since_model_training=1,
            train_model_every=5,
        )
        is True
    )


@pytest.mark.fast
def test_ignores_init_steps_if_offline_training_happened_before() -> None:
    model = EnvironmentModel(1, 1)
    model.has_been_trained_at_least_once = True

    assert (
        model_needs_training(
            model,
            step=1,
            buffer_size=1,
            init_steps=5,
            steps_since_model_training=5,
            train_model_every=1,
        )
        is True
    )


@pytest.mark.fast
def test_false_if_not_using_offline_training_and_init_steps_not_reached() -> None:
    model = EnvironmentModel(1, 1)

    assert (
        model_needs_training(
            model,
            step=1,
            buffer_size=1,
            init_steps=5,
            steps_since_model_training=5,
            train_model_every=1,
        )
        is False
    )


@pytest.mark.fast
def test_returns_false_if_buffer_is_empty() -> None:
    model = EnvironmentModel(1, 1)

    assert (
        model_needs_training(
            model,
            step=1,
            buffer_size=0,
            init_steps=5,
            steps_since_model_training=5,
            train_model_every=1,
        )
        is False
    )


@pytest.mark.fast
def test_trains_model_only_once_during_offline_training() -> None:
    model = EnvironmentModel(1, 1)
    model.has_been_trained_at_least_once = True

    assert (
        model_needs_training(
            model,
            step=-1,
            buffer_size=1,
            init_steps=5,
            steps_since_model_training=1,
            train_model_every=5,
        )
        is False
    )


@pytest.mark.fast
def test_pretrained_model_is_not_trained_before_offline_training() -> None:
    model = EnvironmentModel(1, 1)
    model.has_been_trained_at_least_once = True

    assert (
        model_needs_training(
            model,
            step=-400000,
            buffer_size=400000,
            init_steps=10000,
            steps_since_model_training=1e10,
            train_model_every=250,
        )
        is False
    )
