import argparse
import time
import gym
import torch
from benchmark.models.mlp_actor_critic import MLPActorCritic
from benchmark.sac.sac import sac
from benchmark.utils.run_utils import setup_logger_kwargs
import pytest

logger_kwargs = setup_logger_kwargs('test_sac')
env = 'HalfCheetah-v2'


def test_total_steps_must_be_enough_to_perform_at_least_one_update():
    update_after = 150
    n_steps_per_epoch = 100
    n_epochs = 1

    with pytest.raises(ValueError):
        sac(lambda: gym.make(env), actor_critic=MLPActorCritic, epochs=n_epochs,
            steps_per_epoch=n_steps_per_epoch, update_after=update_after)
        assert True == True


def test_sac_converges_cpu():
    device = 'cpu'
    n_epochs = 5
    start = time.time()

    final_return = sac(lambda: gym.make(env), actor_critic=MLPActorCritic,
                       epochs=n_epochs,
                       logger_kwargs=logger_kwargs, device=device)

    assert final_return

    assert final_return > 500


def test_sac_converges_gpu():
    device = 'cuda'
    n_epochs = 5
    start = time.time()

    final_return = sac(lambda: gym.make(env), actor_critic=MLPActorCritic,
                       epochs=n_epochs,
                       logger_kwargs=logger_kwargs, device=device)

    assert final_return

    assert final_return > 500
