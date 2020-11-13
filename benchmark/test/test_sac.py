import pytest
import torch
from benchmark.train import Trainer
from benchmark.utils.run_utils import setup_logger_kwargs

logger_kwargs = setup_logger_kwargs('test_sac')
env = 'HalfCheetah-v2'


@pytest.mark.fast
def test_total_steps_must_be_enough_to_perform_at_least_one_update():
    init_steps = 150
    n_steps_per_epoch = 100
    n_epochs = 1

    with pytest.raises(ValueError):
        trainer = Trainer(env,
                          epochs=n_epochs,
                          steps_per_epoch=n_steps_per_epoch,
                          init_steps=init_steps,
                          replay_size=5)

        trainer.train()


@pytest.mark.slow
def test_sac_converges():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)

    trainer = Trainer(env,
                      sac_kwargs=dict(hidden=[256, 256, 256, 256],
                                      batch_size=256),
                      random_steps=10000,
                      init_steps=1000,
                      steps_per_epoch=4000,
                      num_test_episodes=10,
                      epochs=5,
                      logger_kwargs=logger_kwargs,
                      device=device)

    final_return, _ = trainer.train()

    assert final_return[-1, -1] > 500
