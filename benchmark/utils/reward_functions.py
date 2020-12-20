from benchmark.utils.mazes import ANTMAZE_MEDIUM_GOAL_BLOCK, ANTMAZE_UMAZE_GOAL_BLOCK
from benchmark.utils.envs import ENV_CATEGORIES


def antmaze_umaze_rew_fn(next_obs, **_):
    x = next_obs[:, :, 0]
    y = next_obs[:, :, 1]

    rewards = 1 * \
        (ANTMAZE_UMAZE_GOAL_BLOCK[0] <= x) * \
        (ANTMAZE_UMAZE_GOAL_BLOCK[1] > x) * \
        (ANTMAZE_UMAZE_GOAL_BLOCK[2] <= y) * \
        (ANTMAZE_UMAZE_GOAL_BLOCK[3] > y)
    rewards = rewards.unsqueeze(-1)

    rewards = rewards*100

    return rewards


def antmaze_medium_rew_fn(next_obs, **_):
    x = next_obs[:, :, 0]
    y = next_obs[:, :, 1]

    rewards = 1 * \
        (ANTMAZE_MEDIUM_GOAL_BLOCK[0] <= x) * \
        (ANTMAZE_MEDIUM_GOAL_BLOCK[1] > x) * \
        (ANTMAZE_MEDIUM_GOAL_BLOCK[2] <= y) * \
        (ANTMAZE_MEDIUM_GOAL_BLOCK[3] > y)
    rewards = rewards.unsqueeze(-1)

    rewards = rewards*100

    return rewards


reward_functions = {
    'antmaze_umaze': antmaze_umaze_rew_fn,
    'antmaze_medium': antmaze_medium_rew_fn
}


def get_reward_function(env_name):
    for fn_name in ENV_CATEGORIES:
        if env_name in ENV_CATEGORIES[fn_name]:
            return reward_functions[fn_name]

    return None
