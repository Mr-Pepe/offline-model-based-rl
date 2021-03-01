from benchmark.utils.postprocessing import get_postprocessing_function
import torch
import gym
import d4rl # noqa
import matplotlib.pyplot as plt
import numpy as np

prefix = 'halfcheetah-'
version = '-v2'

dataset_names = [
    'random',
    'medium-replay',
    'medium',
    'medium-expert',
    'expert',
]

avg_rew = []
per_trajectory_rews = []

for dataset_name in dataset_names:
    env_name = prefix + dataset_name + version
    print(env_name)
    post_fn = get_postprocessing_function(env_name)

    env = gym.make(env_name)

    raw_dataset = env.get_dataset()
    obs = raw_dataset['observations']
    act = raw_dataset['actions']
    obs2 = raw_dataset['next_observations']
    dones = raw_dataset['terminals']
    rew = raw_dataset['rewards']

    avg_rew.append(rew.mean())

    n_timeouts = np.sum(raw_dataset['timeouts']) if 'timeouts' in raw_dataset.keys() else 0
    per_trajectory_rew = np.sum(raw_dataset['rewards']) / (np.sum(raw_dataset['terminals']) + n_timeouts)
    per_trajectory_rews.append(per_trajectory_rew)

    print('\tPer trajectory reward:', per_trajectory_rew)

    for i in range(len(obs)):
        if i % 100 == 0:
            print("{}/{}".format(i, len(obs)), end='\r')
        env.reset()

        env.set_state(
            torch.cat((torch.as_tensor([0]), torch.as_tensor(obs[i][:env.model.nq-1]))), obs[i][env.model.nq-1:])
        real_obs2, r, _, _ = env.step(act[i])

        if abs(r - rew[i]) > 0.01:
            print(abs(r - rew[i]))
            a = 1

        if max(abs(obs2[i] - real_obs2)) > 0.01:
            a = 1

    post_dones = post_fn(torch.as_tensor(obs2).unsqueeze(0))['dones']

    # np.testing.assert_array_equal(dones, post_dones.view(-1))

    print("\tDones in buffer: {}    Postprocessing: {}".format(
        dones.sum(), post_dones.sum()))

for i in range(4):
    assert avg_rew[i] < avg_rew[i+1]
    assert per_trajectory_rews[i] < per_trajectory_rews[i+1]