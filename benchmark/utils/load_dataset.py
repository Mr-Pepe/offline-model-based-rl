import d4rl
import torch
from benchmark.utils.replay_buffer import ReplayBuffer


def load_dataset_from_env(env,
                          n_samples=-1,
                          buffer_size=-1,
                          buffer_device='cpu',
                          with_timeouts=False):
    dataset = d4rl.qlearning_dataset(env)

    if n_samples == -1:
        observations = torch.as_tensor(dataset['observations'])
        next_observations = torch.as_tensor(dataset['next_observations'])
        actions = torch.as_tensor(dataset['actions'])
        rewards = torch.as_tensor(dataset['rewards'])
        dones = torch.as_tensor(dataset['terminals'])

        n_samples = len(dones)
    else:
        observations = torch.as_tensor(dataset['observations'][:n_samples])
        next_observations = torch.as_tensor(
            dataset['next_observations'][:n_samples])
        actions = torch.as_tensor(dataset['actions'][:n_samples])
        rewards = torch.as_tensor(dataset['rewards'][:n_samples])
        dones = torch.as_tensor(dataset['terminals'][:n_samples])

    buffer_size = n_samples if buffer_size == -1 else buffer_size

    buffer: ReplayBuffer = ReplayBuffer(observations.shape[1],
                                        actions.shape[1],
                                        buffer_size,
                                        buffer_device)

    buffer.store_batch(observations,
                       actions,
                       rewards,
                       next_observations,
                       dones)

    if with_timeouts and 'timeouts' in dataset:
        buffer.timeouts = dataset['timeouts']

    return buffer, observations.shape[1], actions.shape[1]
