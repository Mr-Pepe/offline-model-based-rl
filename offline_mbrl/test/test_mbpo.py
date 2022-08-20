import pytest
import torch

from offline_mbrl.train import Trainer
from offline_mbrl.utils.envs import HOPPER_ORIGINAL
from offline_mbrl.utils.run_utils import setup_logger_kwargs


@pytest.mark.slow
def test_mbpo_online():
    logger_kwargs = setup_logger_kwargs("test_mbpo")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        HOPPER_ORIGINAL,
        steps_per_epoch=1000,
        model_kwargs=dict(batch_size=256, type="probabilistic", n_networks=3),
        init_steps=1300,
        random_steps=1000,
        epochs=20,
        use_model=True,
        rollouts_per_step=1,
        logger_kwargs=logger_kwargs,
        device=device,
        seed=0,
    )

    final_return, _ = trainer.train()

    assert final_return[-1, -1] > 170
