from typing import Optional

import d4rl
import gym
import torch

from offline_mbrl.utils.replay_buffer import ReplayBuffer


def load_dataset_from_env(
    env_name: str,
    n_samples: Optional[int] = None,
    buffer_size: Optional[int] = None,
    buffer_device: str = "cpu",
) -> tuple[ReplayBuffer, int, int]:
    """Loads an environment's dataset into a replay buffer.

    Args:
        env_name (str): The environment name.
        n_samples (Optional[int], optional): The number of samples to load from the
            dataset. Pass None to load all samples. Defaults to None.
        buffer_size (Optional[int], optional): The replay buffer size. Pass None to make
            the replay buffer size match the number of loaded samples. Defaults to None.
        buffer_device (str, optional): The device to push the replay buffer content to.
            Defaults to "cpu".

    Returns:
        tuple[ReplayBuffer, int, int]: The replay buffer, the dimensionality of an
            observation, and the dimensionality of an action.
    """
    dataset = d4rl.qlearning_dataset(gym.make(env_name))

    if n_samples is None:
        # Load all samples
        observations = torch.as_tensor(dataset["observations"])
        next_observations = torch.as_tensor(dataset["next_observations"])
        actions = torch.as_tensor(dataset["actions"])
        rewards = torch.as_tensor(dataset["rewards"])
        dones = torch.as_tensor(dataset["terminals"])

        n_samples = len(dones)
    else:
        idxs = torch.randperm(len(dataset["observations"]))[:n_samples]
        observations = torch.as_tensor(dataset["observations"][idxs])
        next_observations = torch.as_tensor(dataset["next_observations"][idxs])
        actions = torch.as_tensor(dataset["actions"][idxs])
        rewards = torch.as_tensor(dataset["rewards"][idxs])
        dones = torch.as_tensor(dataset["terminals"][idxs])

    buffer_size = buffer_size or len(dones)

    buffer: ReplayBuffer = ReplayBuffer(
        observations.shape[1], actions.shape[1], buffer_size, buffer_device
    )

    buffer.store_batch(observations, actions, rewards, next_observations, dones)

    if "timeouts" in dataset:
        buffer.timeouts = dataset["timeouts"]

    return buffer, observations.shape[1], actions.shape[1]
