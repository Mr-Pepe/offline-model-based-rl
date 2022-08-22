import d4rl  # pylint: disable=unused-import
import gym
import pytest
import torch

from offline_mbrl.actors.behavioral_cloning import BC
from offline_mbrl.actors.sac import SAC
from offline_mbrl.train import Trainer
from offline_mbrl.utils.envs import HALF_CHEETAH_EXPERT_V2, HOPPER_MEDIUM_REPLAY_V2
from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.preprocessing import get_preprocessing_function


@pytest.mark.medium
def test_default_agent_is_SAC():
    trainer = Trainer(HOPPER_MEDIUM_REPLAY_V2, pretrain_epochs=0)

    assert isinstance(trainer.agent, SAC)


@pytest.mark.medium
def test_trainer_loads_behavioral_cloning_agent():
    trainer = Trainer(
        HOPPER_MEDIUM_REPLAY_V2, pretrain_epochs=0, agent_kwargs=dict(type="bc")
    )

    assert isinstance(trainer.agent, BC)


@pytest.mark.medium
def test_BC_agent_overfits_on_single_batch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = HALF_CHEETAH_EXPERT_V2
    env = gym.make(env_name)

    buffer, _, _ = load_dataset_from_env(env, buffer_device=device)

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
