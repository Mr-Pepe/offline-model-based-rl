import numpy as np
import pytest

from offline_mbrl.schemas import (
    EnvironmentModelConfiguration,
    SACConfiguration,
    TrainerConfiguration,
)
from offline_mbrl.train import Actions, Trainer
from offline_mbrl.utils.envs import (
    HALF_CHEETAH_MEDIUM_REPLAY_V2,
    HOPPER_MEDIUM_REPLAY_V2,
    HOPPER_RANDOM_V2,
    WALKER_MEDIUM_REPLAY_V2,
)
from offline_mbrl.utils.termination_functions import termination_functions


@pytest.mark.medium
def test_replay_buffer_is_initially_empty_for_online_training() -> None:
    trainer_config = TrainerConfiguration(
        env_name=HOPPER_MEDIUM_REPLAY_V2, offline_epochs=0
    )
    trainer = Trainer(config=trainer_config)
    assert trainer.real_replay_buffer.size == 0


@pytest.mark.medium
def test_replay_buffer_is_filled_for_offline_training() -> None:
    trainer_config = TrainerConfiguration(
        env_name=HOPPER_MEDIUM_REPLAY_V2, offline_epochs=1, n_samples_from_dataset=100
    )
    trainer = Trainer(config=trainer_config)

    assert trainer.real_replay_buffer.size == 100


@pytest.mark.medium
def test_actions_for_online_model_free_training() -> None:
    trainer_config = TrainerConfiguration(
        env_name=HOPPER_MEDIUM_REPLAY_V2,
        online_epochs=3,
        steps_per_epoch=50,
        init_steps=100,
        random_steps=50,
        max_episode_length=30,
    )

    agent_config = SACConfiguration(
        hidden_layer_sizes=(32, 32, 32, 32), training_batch_size=32
    )

    trainer = Trainer(config=trainer_config, agent_config=agent_config)

    test_performances, action_log = trainer.train()

    np.testing.assert_array_equal(
        test_performances.shape, (trainer_config.online_epochs, 2)
    )

    # Only the last epoch performs test runs
    assert test_performances[0][0] == 1
    assert test_performances[0][1] == 0
    assert test_performances[1][0] == 2
    assert test_performances[1][1] == 0
    assert test_performances[2][0] == 3
    assert test_performances[2][1] != 0

    total_steps = trainer_config.online_epochs * trainer_config.steps_per_epoch

    np.testing.assert_array_equal(action_log.shape, (total_steps, len(Actions)))

    np.testing.assert_array_equal(action_log[:, Actions.TRAIN_MODEL], [0] * total_steps)
    np.testing.assert_array_equal(
        action_log[:, Actions.UPDATE_AGENT],
        [0] * trainer_config.init_steps
        + [1] * (total_steps - trainer_config.init_steps),
    )
    np.testing.assert_array_equal(
        action_log[:, Actions.RANDOM_ACTION],
        [1] * trainer_config.random_steps
        + [0] * (total_steps - trainer_config.random_steps),
    )
    np.testing.assert_array_equal(
        action_log[:, Actions.GENERATE_ROLLOUTS], [0] * total_steps
    )
    np.testing.assert_array_equal(
        action_log[:, Actions.INTERACT_WITH_ENV], [1] * total_steps
    )


@pytest.mark.medium
def test_actions_for_online_model_based_training() -> None:
    trainer_config = TrainerConfiguration(
        env_name=HOPPER_RANDOM_V2,
        online_epochs=5,
        steps_per_epoch=100,
        init_steps=300,
        random_steps=50,
        train_env_model_every=50,
        use_env_model=True,
        max_episode_length=30,
    )

    agent_config = SACConfiguration(
        hidden_layer_sizes=(32, 32, 32), training_batch_size=32
    )

    env_model_config = EnvironmentModelConfiguration(
        hidden_layer_sizes=(32, 32), training_batch_size=32, training_patience=1
    )

    trainer = Trainer(
        config=trainer_config,
        agent_config=agent_config,
        env_model_config=env_model_config,
    )

    test_performances, action_log = trainer.train()

    np.testing.assert_array_equal(
        test_performances.shape, (trainer_config.online_epochs, 2)
    )

    # Only the last two epoch perform test runs
    assert test_performances[0][0] == 1
    assert test_performances[0][1] == 0
    assert test_performances[1][0] == 2
    assert test_performances[1][1] == 0
    assert test_performances[2][0] == 3
    assert test_performances[2][1] == 0
    assert test_performances[3][0] == 4
    assert test_performances[3][1] != 0
    assert test_performances[4][0] == 5
    assert test_performances[4][1] != 0

    total_steps = trainer_config.online_epochs * trainer_config.steps_per_epoch

    np.testing.assert_array_equal(action_log.shape, (total_steps, len(Actions)))

    np.testing.assert_array_equal(
        action_log[:, Actions.TRAIN_MODEL],
        [0] * (trainer_config.init_steps)
        + [1]
        + ([0] * (trainer_config.train_env_model_every - 1) + [1]) * 3
        + [0]
        * (
            total_steps
            - trainer_config.init_steps
            - 3 * trainer_config.train_env_model_every
            - 1
        ),
    )
    np.testing.assert_array_equal(
        action_log[:, Actions.UPDATE_AGENT],
        [0] * trainer_config.init_steps
        + [1] * (total_steps - trainer_config.init_steps),
    )
    np.testing.assert_array_equal(
        action_log[:, Actions.RANDOM_ACTION],
        [1] * trainer_config.random_steps
        + [0] * (total_steps - trainer_config.random_steps),
    )
    np.testing.assert_array_equal(
        action_log[:, Actions.GENERATE_ROLLOUTS],
        [0] * trainer_config.init_steps
        + [1] * (total_steps - trainer_config.init_steps),
    )
    np.testing.assert_array_equal(
        action_log[:, Actions.INTERACT_WITH_ENV], [1] * total_steps
    )


@pytest.mark.medium
def test_actions_for_offline_model_free_training_with_fine_tuning() -> None:
    trainer_config = TrainerConfiguration(
        env_name=HOPPER_MEDIUM_REPLAY_V2,
        online_epochs=3,
        offline_epochs=3,
        steps_per_epoch=50,
        init_steps=100,
        random_steps=50,
        n_samples_from_dataset=500_000,
        max_episode_length=30,
    )

    agent_config = SACConfiguration(
        hidden_layer_sizes=(32, 32, 32), training_batch_size=32
    )

    trainer = Trainer(config=trainer_config, agent_config=agent_config)

    test_performances, action_log = trainer.train()

    np.testing.assert_array_equal(
        test_performances.shape,
        (trainer_config.online_epochs + trainer_config.offline_epochs, 2),
    )

    for epoch, performance in enumerate(test_performances):
        assert performance[0] == epoch + 1 - trainer_config.offline_epochs
        assert performance[1] != 0

    total_steps = (
        trainer_config.online_epochs * trainer_config.steps_per_epoch
        + trainer_config.offline_epochs * trainer_config.steps_per_epoch
    )

    np.testing.assert_array_equal(action_log.shape, (total_steps, len(Actions)))

    np.testing.assert_array_equal(action_log[:, Actions.TRAIN_MODEL], [0] * total_steps)
    np.testing.assert_array_equal(
        action_log[:, Actions.UPDATE_AGENT], [1] * total_steps
    )
    np.testing.assert_array_equal(
        action_log[:, Actions.RANDOM_ACTION], [0] * total_steps
    )
    np.testing.assert_array_equal(
        action_log[:, Actions.GENERATE_ROLLOUTS], [0] * total_steps
    )
    np.testing.assert_array_equal(
        action_log[:, Actions.INTERACT_WITH_ENV],
        [0] * trainer_config.steps_per_epoch * trainer_config.offline_epochs
        + [1] * trainer_config.steps_per_epoch * trainer_config.online_epochs,
    )


@pytest.mark.slow
def test_actions_for_offline_model_based_training_with_fine_tuning() -> None:
    trainer_config = TrainerConfiguration(
        env_name=HOPPER_RANDOM_V2,
        online_epochs=5,
        offline_epochs=3,
        steps_per_epoch=100,
        init_steps=300,
        random_steps=50,
        train_env_model_every=50,
        use_env_model=True,
        max_episode_length=30,
        n_parallel_virtual_rollouts=1,
        n_samples_from_dataset=None,
    )

    agent_config = SACConfiguration(
        hidden_layer_sizes=(32, 32, 32, 32), training_batch_size=32
    )

    env_model_config = EnvironmentModelConfiguration(
        hidden_layer_sizes=(32, 32), training_batch_size=32, training_patience=0
    )

    trainer = Trainer(
        config=trainer_config,
        agent_config=agent_config,
        env_model_config=env_model_config,
    )

    test_performances, action_log = trainer.train()

    np.testing.assert_array_equal(
        test_performances.shape,
        (trainer_config.online_epochs + trainer_config.offline_epochs, 2),
    )

    for epoch, performance in enumerate(test_performances):
        assert performance[0] == epoch + 1 - trainer_config.offline_epochs
        assert performance[1] != 0

    total_steps = (
        trainer_config.online_epochs * trainer_config.steps_per_epoch
        + trainer_config.offline_epochs * trainer_config.steps_per_epoch
    )

    np.testing.assert_array_equal(action_log.shape, (total_steps, len(Actions)))

    np.testing.assert_array_equal(
        action_log[:, Actions.TRAIN_MODEL],
        [1]
        + [0] * (trainer_config.offline_epochs * trainer_config.steps_per_epoch)
        + ([0] * (trainer_config.train_env_model_every - 1) + [1]) * 9
        + [0] * (trainer_config.train_env_model_every - 1),
    )
    np.testing.assert_array_equal(
        action_log[:, Actions.UPDATE_AGENT], [1] * total_steps
    )
    np.testing.assert_array_equal(
        action_log[:, Actions.RANDOM_ACTION],
        [1] * trainer_config.random_steps
        + [0] * (total_steps - trainer_config.random_steps),
    )
    np.testing.assert_array_equal(
        action_log[:, Actions.GENERATE_ROLLOUTS], [1] * total_steps
    )
    np.testing.assert_array_equal(
        action_log[:, Actions.INTERACT_WITH_ENV],
        [0] * trainer_config.steps_per_epoch * trainer_config.offline_epochs
        + [1] * trainer_config.steps_per_epoch * trainer_config.online_epochs,
    )


@pytest.mark.medium
def test_trainer_picks_correct_termination_functions() -> None:
    # pylint: disable=comparison-with-callable

    trainer = Trainer(
        config=TrainerConfiguration(
            env_name=HOPPER_MEDIUM_REPLAY_V2, use_env_model=True
        )
    )

    assert (
        trainer.env_model_config.termination_function == termination_functions["hopper"]
    )

    trainer = Trainer(
        config=TrainerConfiguration(
            env_name=HALF_CHEETAH_MEDIUM_REPLAY_V2, use_env_model=True
        )
    )

    assert (
        trainer.env_model_config.termination_function
        == termination_functions["half_cheetah"]
    )

    trainer = Trainer(
        config=TrainerConfiguration(
            env_name=WALKER_MEDIUM_REPLAY_V2, use_env_model=True
        )
    )

    assert (
        trainer.env_model_config.termination_function
        == termination_functions["walker2d"]
    )


@pytest.mark.slow
def test_results_are_reproducible() -> None:
    trainer_config = TrainerConfiguration(
        env_name=HOPPER_RANDOM_V2,
        online_epochs=5,
        steps_per_epoch=100,
        init_steps=300,
        random_steps=50,
        train_env_model_every=50,
        use_env_model=True,
        max_episode_length=30,
        seed=1,
    )

    agent_config = SACConfiguration(
        hidden_layer_sizes=(32, 32, 32, 32), training_batch_size=32
    )

    env_model_config = EnvironmentModelConfiguration(
        hidden_layer_sizes=(32, 32), training_batch_size=32, training_patience=1
    )

    trainer1 = Trainer(
        config=trainer_config,
        agent_config=agent_config,
        env_model_config=env_model_config,
    )

    test_performances1, action_log1 = trainer1.train()

    trainer2 = Trainer(
        config=trainer_config,
        agent_config=agent_config,
        env_model_config=env_model_config,
    )

    test_performances2, action_log2 = trainer2.train()

    np.testing.assert_array_equal(test_performances1, test_performances2)
    np.testing.assert_array_equal(action_log1, action_log2)
