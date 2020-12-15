from benchmark.utils.logx import EpochLogger
import time
import gym
from benchmark.actors.sac import SAC
from benchmark.utils.load_dataset import load_dataset_from_env
import pytest
import torch
from benchmark.train import Trainer
from benchmark.utils.run_utils import setup_logger_kwargs
import d4rl  # noqa


@pytest.mark.fast
def test_total_steps_must_be_enough_to_perform_at_least_one_update():
    init_steps = 150
    n_steps_per_epoch = 100
    n_epochs = 1
    env = 'HalfCheetah-v2'

    with pytest.raises(ValueError):
        trainer = Trainer(env,
                          epochs=n_epochs,
                          steps_per_epoch=n_steps_per_epoch,
                          init_steps=init_steps,
                          real_buffer_size=5)

        trainer.train()


@pytest.mark.slow
def test_sac_converges():
    logger_kwargs = setup_logger_kwargs('test_sac')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = 'HalfCheetah-v2'

    trainer = Trainer(env,
                      sac_kwargs=dict(hidden=[256, 256, 256, 256],
                                      batch_size=256),
                      random_steps=10000,
                      init_steps=1000,
                      steps_per_epoch=4000,
                      num_test_episodes=10,
                      epochs=5,
                      logger_kwargs=logger_kwargs,
                      device=device,
                      seed=0)

    final_return, _ = trainer.train()

    assert final_return[-1, -1] > 400


@pytest.mark.slow
def test_sac_offline():
    logger_kwargs = setup_logger_kwargs('test_sac_offline')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = 'halfcheetah-random-v0'
    n_samples = 900000

    trainer = Trainer(env,
                      sac_kwargs=dict(hidden=[200, 200, 200, 200],
                                      batch_size=256),
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
                      seed=0)

    final_return, _ = trainer.train()

    assert final_return[-1, -1] > 400


def sac_trains_faster_on_gpu_on_filled_buffer():

    for device in ['cpu', 'cuda']:
        torch.manual_seed(0)

        env = gym.make('halfcheetah-random-v0')
        buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                         buffer_device=device,
                                                         n_samples=100000)

        agent = SAC(
            env.observation_space,
            env.action_space,
            batch_size=256,
            device=device
        )

        logger = EpochLogger(**logger_kwargs)

        n_updates = 50
        i_update = 0

        start_time = time.time()

        for _ in range(n_updates):
            print("Update {}/{}".format(i_update, n_updates), end='\r')
            agent.multi_update(20, buffer, logger)
            i_update += 1

        print('')
        print("Time: {}".format(time.time()-start_time))
