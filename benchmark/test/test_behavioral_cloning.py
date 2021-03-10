import gym
from benchmark.actors.behavioral_cloning import BC
from benchmark.actors.sac import SAC
import pytest
from benchmark.utils.envs import HALF_CHEETAH_EXPERT_V2, HOPPER_EXPERT_V2, HOPPER_MEDIUM_REPLAY_V2
from benchmark.utils.preprocessing import get_preprocessing_function
from benchmark.utils.load_dataset import load_dataset_from_env
from benchmark.train import Trainer
import torch
import d4rl  # noqa


@pytest.mark.medium
def test_default_agent_is_SAC():
    trainer = Trainer(HOPPER_MEDIUM_REPLAY_V2,
                      pretrain_epochs=0)

    assert type(trainer.agent) is SAC


@pytest.mark.medium
def test_trainer_loads_behavioral_cloning_agent():
    trainer = Trainer(HOPPER_MEDIUM_REPLAY_V2,
                      pretrain_epochs=0,
                      agent_kwargs=dict(
                          type='bc'
                      ))

    assert type(trainer.agent) is BC


@pytest.mark.medium
def test_BC_agent_overfits_on_single_batch():
    env_name = HALF_CHEETAH_EXPERT_V2
    env = gym.make(env_name)

    buffer, obs_dim, act_dim = load_dataset_from_env(env)

    agent = BC(env.observation_space, env.action_space, batch_size=256,
               lr=1e-5,
               pre_fn=get_preprocessing_function(env_name))
    batch = buffer.sample_batch(256)

    for i in range(1000000):
        loss = agent.update(batch)

        if i % 100 == 0:
            print(loss, end='\r')


@pytest.mark.medium
def test_train_BC_agent():
    epochs = 100
    steps_per_epoch = 10000
    init_steps = 100
    random_steps = 50

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(HALF_CHEETAH_EXPERT_V2,
                      epochs=epochs,
                      steps_per_epoch=steps_per_epoch,
                      max_ep_len=30,
                      init_steps=init_steps,
                      random_steps=random_steps,
                      agent_kwargs=dict(
                          type='bc',
                          batch_size=256
                      ),
                      use_model=False,
                      env_steps_per_step=0,
                      n_samples_from_dataset=-1,
                      device='cpu')

    test_performances, action_log = trainer.train()
