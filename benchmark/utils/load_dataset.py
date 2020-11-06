import d4rl
from benchmark.utils.replay_buffer import ReplayBuffer


def load_dataset_from_env(env,
                          n_samples=-1,
                          buffer_size=-1,
                          buffer_device='cpu'):
    dataset = d4rl.qlearning_dataset(env)
    observations = dataset['observations']
    next_observations = dataset['next_observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    dones = dataset['terminals']

    n_samples = len(dataset['observations']) if n_samples == -1 else n_samples
    buffer_size = n_samples if buffer_size == -1 else buffer_size

    buffer = ReplayBuffer(observations.shape[1],
                          actions.shape[1],
                          buffer_size,
                          buffer_device)

    buffer.store_batch(observations[:n_samples],
                       actions[:n_samples],
                       rewards[:n_samples],
                       next_observations[:n_samples],
                       dones[:n_samples])

    return buffer, observations.shape[1], actions.shape[1]
