import pytest
import torch

from offline_mbrl.schemas import (
    EpochLoggerConfiguration,
    SACConfiguration,
    TrainerConfiguration,
)
from offline_mbrl.train import Trainer
from offline_mbrl.utils.envs import HALF_CHEETAH_RANDOM_V2
from offline_mbrl.utils.preprocessing import get_preprocessing_function


@pytest.mark.fast
def test_total_steps_must_be_enough_to_perform_at_least_one_update() -> None:
    trainer_config = TrainerConfiguration(
        env_name=HALF_CHEETAH_RANDOM_V2,
        init_steps=150,
        steps_per_epoch=100,
        online_epochs=1,
        real_buffer_size=5,
    )

    with pytest.raises(
        ValueError, match="Number of total steps lower than init steps."
    ):
        trainer = Trainer(trainer_config)
        trainer.train()


@pytest.mark.slow
def test_sac_converges() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer_config = TrainerConfiguration(
        env_name=HALF_CHEETAH_RANDOM_V2,
        random_steps=10_000,
        init_steps=1_000,
        steps_per_epoch=4_000,
        test_episodes=10,
        online_epochs=5,
        device=device,
        seed=0,
    )

    agent_config = SACConfiguration(
        hidden_layer_sizes=(32, 32, 32),
        training_batch_size=256,
        preprocessing_function=get_preprocessing_function(HALF_CHEETAH_RANDOM_V2),
    )

    logger_config = EpochLoggerConfiguration(output_dir="test_sac")

    trainer = Trainer(
        config=trainer_config,
        agent_config=agent_config,
        logger_config=logger_config,
    )

    final_return, _ = trainer.train()

    assert final_return[-1, -1] > 400


@pytest.mark.slow
def test_sac_offline() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer_config = TrainerConfiguration(
        env_name=HALF_CHEETAH_RANDOM_V2,
        n_samples_from_dataset=900_000,
        random_steps=0,
        agent_updates_per_step=1,
        render_test_episodes=True,
        init_steps=0,
        steps_per_epoch=4_000,
        test_episodes=10,
        online_epochs=0,
        offline_epochs=4,
        device=device,
        seed=0,
    )

    agent_config = SACConfiguration(
        hidden_layer_sizes=(32, 32, 32),
        training_batch_size=256,
        preprocessing_function=get_preprocessing_function(HALF_CHEETAH_RANDOM_V2),
    )

    logger_config = EpochLoggerConfiguration(output_dir="test_sac_offline")

    trainer = Trainer(
        config=trainer_config,
        agent_config=agent_config,
        logger_config=logger_config,
    )

    final_return, _ = trainer.train()

    assert final_return[-1, -1] > 1000
