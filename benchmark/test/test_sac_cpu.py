import argparse

import gym
import torch
from benchmark.models.mlp_actor_critic import MLPActorCritic
from benchmark.sac.sac import sac
from benchmark.utils.run_utils import setup_logger_kwargs

logger_kwargs = setup_logger_kwargs('test_sac_cpu')

def test_sac_cpu():
    sac(lambda: gym.make('HalfCheetah-v2'), actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[3, 3]), epochs=1,
        logger_kwargs=logger_kwargs, device='cpu')

    assert True == True
