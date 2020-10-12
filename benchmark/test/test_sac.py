import argparse
import time
import gym
import torch
from benchmark.models.mlp_actor_critic import MLPActorCritic
from benchmark.sac.sac import sac
from benchmark.utils.run_utils import setup_logger_kwargs
import pytest

logger_kwargs = setup_logger_kwargs('test_sac_cpu')
device = 'cpu'
env = 'HalfCheetah-v2'
n_epochs = 1
hidden_sizes = [3, 3]

def test_total_steps_must_be_enough_to_perform_at_least_one_update():
    update_after = 150
    n_steps_per_epoch = 100

    with pytest.raises(ValueError):
        sac(lambda: gym.make(env), actor_critic=MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=hidden_sizes), epochs=n_epochs,
            steps_per_epoch=n_steps_per_epoch, update_after=update_after,
            logger_kwargs=logger_kwargs, device='cpu')
        assert True == True


# def test_sac_cpu():

#     n_steps_per_epoch = 1000
#     total_steps = n_steps_per_epoch * n_epochs

#     start = time.time()

#     sac(lambda: gym.make(env), actor_critic=MLPActorCritic,
#         ac_kwargs=dict(hidden_sizes=hidden_sizes), epochs=n_epochs, steps_per_epoch=n_steps_per_epoch,
#         logger_kwargs=logger_kwargs, device='cpu')

#     assert True == True, 'Test {} on {} for {} steps with {} layers in {}s'.format(
#         env, device, total_steps, hidden_sizes, time.time())


# def test_sac_gpu():
#     sac(lambda: gym.make('HalfCheetah-v2'), actor_critic=MLPActorCritic,
#         ac_kwargs=dict(hidden_sizes=[3, 3]), epochs=1, steps_per_epoch=100,
#         logger_kwargs=logger_kwargs, device='cuda')

#     assert True == True
