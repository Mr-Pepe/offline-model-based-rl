import gym
import torch
from benchmark.train import train
from benchmark.utils.run_utils import setup_logger_kwargs

logger_kwargs = setup_logger_kwargs('test_mbpo')
env = 'HalfCheetah-v2'


def test_mbpo_with_single_deterministic_model_converges():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs = 5

    final_return = train(lambda: gym.make(env),
                         epochs=n_epochs,
                         logger_kwargs=logger_kwargs,
                         device=device, use_model=True, steps_per_epoch=1200)

    assert final_return

    assert final_return > 500
