import d4rl  # noqa
import gym
import pytest
import torch
from offline_mbrl.actors.behavioral_cloning import BC
from offline_mbrl.actors.sac import SAC
from offline_mbrl.train import Trainer
from offline_mbrl.utils.envs import (
    HALF_CHEETAH_EXPERT_V2,
    HOPPER_EXPERT_V2,
    HOPPER_MEDIUM_REPLAY_V2,
)
from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.preprocessing import get_preprocessing_function


@pytest.mark.medium
def test_default_agent_is_SAC():
    trainer = Trainer(HOPPER_MEDIUM_REPLAY_V2, pretrain_epochs=0)

    assert type(trainer.agent) is SAC


@pytest.mark.medium
def test_trainer_loads_behavioral_cloning_agent():
    trainer = Trainer(
        HOPPER_MEDIUM_REPLAY_V2, pretrain_epochs=0, agent_kwargs=dict(type="bc")
    )

    assert type(trainer.agent) is BC


@pytest.mark.medium
def test_BC_agent_overfits_on_single_batch():
    env_name = HALF_CHEETAH_EXPERT_V2
    env = gym.make(env_name)

    buffer, obs_dim, act_dim = load_dataset_from_env(env)

    agent = BC(
        env.observation_space,
        env.action_space,
        batch_size=256,
        lr=1e-4,
        pre_fn=get_preprocessing_function(env_name),
    )
    batch = buffer.sample_batch(256)

    for i in range(5000):
        loss = agent.update(batch)

        if i % 100 == 0:
            print(loss, end="\r")

    assert loss < 1e-5


def test_BC_agent_trains_on_dataset():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = HALF_CHEETAH_EXPERT_V2
    env = gym.make(env_name)

    buffer, obs_dim, act_dim = load_dataset_from_env(
        env, n_samples=-1, buffer_device=device
    )

    agent = BC(
        env.observation_space,
        env.action_space,
        batch_size=256,
        lr=1e-4,
        pre_fn=get_preprocessing_function(env_name, device=device),
        device=device,
    )

    for i in range(100):
        loss = agent.multi_update(1, buffer, debug=True)

        if i % 100 == 0:
            print(loss, end="\r")


def test_train_BC_agent():
    epochs = 100
    steps_per_epoch = 10000
    init_steps = 100
    random_steps = 50

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        HALF_CHEETAH_EXPERT_V2,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        max_ep_len=30,
        init_steps=init_steps,
        random_steps=random_steps,
        agent_kwargs=dict(type="bc", batch_size=256),
        use_model=False,
        env_steps_per_step=0,
        n_samples_from_dataset=-1,
        device=device,
    )

    test_performances, action_log = trainer.train()
