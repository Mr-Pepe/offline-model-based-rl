import gym
import torch
from benchmark.train import Trainer
from benchmark.utils.run_utils import setup_logger_kwargs
import pytest


@pytest.mark.slow
def test_mbpo_with_single_deterministic_model_converges():
    logger_kwargs = setup_logger_kwargs('test_mbpo')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(lambda: gym.make('Hopper-v2'),
                      term_fn='hopper',
                      steps_per_epoch=1000,
                      random_steps=1000,
                      epochs=20,
                      use_model=True,
                      model_rollouts=1,
                      logger_kwargs=logger_kwargs,
                      device=device)

    final_return = trainer.train()

    assert final_return

    assert final_return > 200
