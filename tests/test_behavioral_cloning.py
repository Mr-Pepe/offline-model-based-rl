# pylint: disable=unused-import
import gym
import pytest
import torch

from offline_mbrl.actors.behavioral_cloning import BehavioralCloningAgent
from offline_mbrl.actors.sac import SAC
from offline_mbrl.schemas import BehavioralCloningConfiguration, TrainerConfiguration
from offline_mbrl.train import Trainer
from offline_mbrl.utils.envs import (
    HALF_CHEETAH_MEDIUM_REPLAY_V2,
    HOPPER_MEDIUM_REPLAY_V2,
)
from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.preprocessing import get_preprocessing_function


@pytest.mark.medium
def test_default_agent_is_SAC() -> None:
    trainer_config = TrainerConfiguration(
        env_name=HOPPER_MEDIUM_REPLAY_V2, offline_epochs=0
    )
    trainer = Trainer(config=trainer_config)

    assert isinstance(trainer.agent, SAC)


@pytest.mark.medium
def test_trainer_loads_behavioral_cloning_agent() -> None:
    trainer_config = TrainerConfiguration(
        env_name=HOPPER_MEDIUM_REPLAY_V2, offline_epochs=0
    )
    agent_config = BehavioralCloningConfiguration(type="bc")
    trainer = Trainer(config=trainer_config, agent_config=agent_config)

    assert isinstance(trainer.agent, BehavioralCloningAgent)


@pytest.mark.medium
def test_BC_agent_overfits_on_single_batch() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = HOPPER_MEDIUM_REPLAY_V2
    env = gym.make(env_name)

    buffer, _, _ = load_dataset_from_env(env_name=env_name, buffer_device=device)

    agent = BehavioralCloningAgent(
        env.observation_space,
        env.action_space,
        BehavioralCloningConfiguration(
            training_batch_size=256,
            lr=1e-4,
            preprocessing_function=get_preprocessing_function(env_name),
        ),
    )
    batch = buffer.sample_batch(256)

    for i in range(5000):
        loss = agent.update(batch)

        if i % 100 == 0:
            print(loss, end="\r")

    assert loss < 1e-5
