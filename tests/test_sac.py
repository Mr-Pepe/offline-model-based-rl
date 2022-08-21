import pytest
import torch

from offline_mbrl.train import Trainer
from offline_mbrl.utils.envs import HALF_CHEETAH_RANDOM
from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.logx import EpochLogger
from offline_mbrl.utils.run_utils import setup_logger_kwargs


@pytest.mark.fast
def test_total_steps_must_be_enough_to_perform_at_least_one_update():
    init_steps = 150
    n_steps_per_epoch = 100
    n_epochs = 1
    env = "HalfCheetah-v2"

    with pytest.raises(ValueError):
        trainer = Trainer(
            env,
            epochs=n_epochs,
            steps_per_epoch=n_steps_per_epoch,
            init_steps=init_steps,
            real_buffer_size=5,
        )

        trainer.train()


@pytest.mark.slow
def test_sac_converges():
    logger_kwargs = setup_logger_kwargs("test_sac")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = "HalfCheetah-v2"

    trainer = Trainer(
        env,
        agent_kwargs=dict(hidden=[256, 256, 256, 256], batch_size=256),
        random_steps=10000,
        init_steps=1000,
        steps_per_epoch=4000,
        num_test_episodes=10,
        epochs=5,
        logger_kwargs=logger_kwargs,
        device=device,
        seed=0,
    )

    final_return, _ = trainer.train()

    assert final_return[-1, -1] > 400


@pytest.mark.slow
def test_sac_offline():
    logger_kwargs = setup_logger_kwargs("test_sac_offline")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = HALF_CHEETAH_RANDOM
    n_samples = 900000

    trainer = Trainer(
        env,
        agent_kwargs=dict(hidden=[200, 200, 200, 200], batch_size=256),
        random_steps=0,
        agent_updates_per_step=1,
        render=True,
        init_steps=0,
        n_samples_from_dataset=n_samples,
        steps_per_epoch=1000,
        num_test_episodes=10,
        epochs=0,
        pretrain_epochs=20,
        logger_kwargs=logger_kwargs,
        device=device,
        seed=0,
    )

    final_return, _ = trainer.train()

    assert final_return[-1, -1] > 400
