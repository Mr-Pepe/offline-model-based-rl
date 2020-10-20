import d4rl
from benchmark.utils.replay_buffer import ReplayBuffer


def load_dataset_from_env(env, n_samples=-1):
    dataset = d4rl.qlearning_dataset(env)
    observations = dataset['observations']
    actions = dataset['actions']

    buffer = ReplayBuffer(observations.shape[1], actions.shape[1], 1000000)

    n_samples = len(dataset['observations']) if n_samples == -1 else n_samples

    for i in range(n_samples):
        buffer.store(dataset['observations'][i],
                     dataset['actions'][i],
                     dataset['rewards'][i],
                     dataset['next_observations'][i],
                     dataset['terminals'][i])

    return buffer, observations.shape[1], actions.shape[1]
