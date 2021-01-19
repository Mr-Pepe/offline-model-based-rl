from benchmark.utils.mazes import ANTMAZE_ANT_RADIUS, ANTMAZE_MEDIUM_DIVERSE_GOAL, ANTMAZE_UMAZE_DIVERSE_GOAL, ANTMAZE_UMAZE_GOAL
import torch


def antmaze_umaze_rew_fn(next_obs, **_):
    x = next_obs[:, :, 0]
    y = next_obs[:, :, 1]

    rewards = torch.sqrt(torch.square(x - ANTMAZE_UMAZE_GOAL[0]) +
                         torch.square(y - ANTMAZE_UMAZE_GOAL[1])) < ANTMAZE_ANT_RADIUS
    rewards = rewards.unsqueeze(-1)

    rewards = rewards*100

    return rewards


def antmaze_umaze_diverse_rew_fn(next_obs, **_):
    x = next_obs[:, :, 0]
    y = next_obs[:, :, 1]

    rewards = torch.sqrt(torch.square(x - ANTMAZE_UMAZE_DIVERSE_GOAL[0]) +
                         torch.square(y - ANTMAZE_UMAZE_DIVERSE_GOAL[1])) < ANTMAZE_ANT_RADIUS
    rewards = rewards.unsqueeze(-1)

    rewards = rewards*100

    return rewards


def antmaze_medium_diverse_rew_fn(next_obs, **_):
    x = next_obs[:, :, 0]
    y = next_obs[:, :, 1]

    rewards = torch.sqrt(torch.square(x - ANTMAZE_MEDIUM_DIVERSE_GOAL[0]) +
                         torch.square(y - ANTMAZE_MEDIUM_DIVERSE_GOAL[1])) < ANTMAZE_ANT_RADIUS
    rewards = rewards.unsqueeze(-1)

    rewards = rewards*100

    return rewards


reward_functions = {
    'antmaze-umaze-v0': antmaze_umaze_rew_fn,
    'antmaze-umaze-diverse-v0': antmaze_umaze_diverse_rew_fn,
    'antmaze-medium-diverse-v0': antmaze_medium_diverse_rew_fn
}


def get_reward_function(env_name):
    if env_name in reward_functions:
        return reward_functions[env_name]

    return None
