import gym
import pytest
import torch
from benchmark.train import train
from benchmark.utils.run_utils import setup_logger_kwargs

logger_kwargs = setup_logger_kwargs('test_sac')
env = 'HalfCheetah-v2'


def test_total_steps_must_be_enough_to_perform_at_least_one_update():
    init_steps = 150
    n_steps_per_epoch = 100
    n_epochs = 1

    with pytest.raises(ValueError):
        train(lambda: gym.make(env),
              epochs=n_epochs,
              steps_per_epoch=n_steps_per_epoch,
              init_steps=init_steps)
        assert True == True


def test_sac_converges_cpu():
    device = 'cpu'
    n_epochs = 5

    final_return = train(lambda: gym.make(env),
                         epochs=n_epochs,
                         logger_kwargs=logger_kwargs,
                         device=device)

    assert final_return

    assert final_return > 500


def test_sac_converges_gpu():
    device = 'cuda'
    n_epochs = 5

    assert torch.cuda.is_available()

    final_return = train(lambda: gym.make(env),
                         epochs=n_epochs,
                         logger_kwargs=logger_kwargs,
                         device=device)

    assert final_return

    assert final_return > 500
