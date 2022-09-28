import pytest
import torch

from offline_mbrl.schemas import (
    EnvironmentModelConfiguration,
    EpochLoggerConfiguration,
    TrainerConfiguration,
)
from offline_mbrl.train import Trainer
from offline_mbrl.utils.envs import HOPPER_MEDIUM_REPLAY_V2


@pytest.mark.slow
def test_mbpo_online() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer_config = TrainerConfiguration(
        env_name=HOPPER_MEDIUM_REPLAY_V2,
        steps_per_epoch=4000,
        init_steps=3000,
        random_steps=3000,
        online_epochs=10,
        use_env_model=True,
        n_parallel_virtual_rollouts=1,
        device=device,
        seed=0,
    )

    model_config = EnvironmentModelConfiguration(
        type="probabilistic", n_networks=3, training_batch_size=256
    )

    logger_config = EpochLoggerConfiguration(output_dir="test_mbpo")

    trainer = Trainer(
        config=trainer_config,
        env_model_config=model_config,
        logger_config=logger_config,
    )

    final_return, _ = trainer.train()

    assert final_return[-1, -1] > 150
