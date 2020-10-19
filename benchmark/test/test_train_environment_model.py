import d4rl
import gym
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.replay_buffer import ReplayBuffer
from benchmark.utils.train_environment_model import train_environment_model
import pytest

def test_train_environment_model():
    env = gym.make('halfcheetah-random-v0')
    dataset = d4rl.qlearning_dataset(env)
    observations = dataset['observations']
    actions = dataset['actions']

    buffer = ReplayBuffer(observations.shape[1], actions.shape[1], 1000000)

    for i in range(len(dataset['observations'])):
        buffer.store(dataset['observations'][i],
        dataset['actions'][i],
        dataset['rewards'][i],
        dataset['next_observations'][i],
        dataset['terminals'][i])

    model = EnvironmentModel(observations.shape[1], actions.shape[1])


    loss = train_environment_model(model, buffer, val_split=0.2, patience=10)

    assert loss < 0.3

def test_raise_error_if_data_not_enough_for_split_at_given_batch_size():

    env = gym.make('halfcheetah-random-v0')
    dataset = d4rl.qlearning_dataset(env)
    observations = dataset['observations']
    actions = dataset['actions']

    buffer = ReplayBuffer(observations.shape[1], actions.shape[1], 1000000)

    for i in range(100):
        buffer.store(dataset['observations'][i],
        dataset['actions'][i],
        dataset['rewards'][i],
        dataset['next_observations'][i],
        dataset['terminals'][i])

    with pytest.raises(ValueError):
        model = EnvironmentModel(observations.shape[1], actions.shape[1])

        train_environment_model(model, buffer, val_split=0.2, patience=10)