from functools import partial
from benchmark.utils.envs import HALF_CHEETAH_EXPERT, HALF_CHEETAH_EXPERT_V1, HALF_CHEETAH_EXPERT_V2, HALF_CHEETAH_MEDIUM, HALF_CHEETAH_MEDIUM_EXPERT, HALF_CHEETAH_MEDIUM_EXPERT_V1, HALF_CHEETAH_MEDIUM_EXPERT_V2, HALF_CHEETAH_MEDIUM_REPLAY, HALF_CHEETAH_MEDIUM_REPLAY_V1, HALF_CHEETAH_MEDIUM_REPLAY_V2, HALF_CHEETAH_MEDIUM_V1, HALF_CHEETAH_MEDIUM_V2, HALF_CHEETAH_RANDOM, HALF_CHEETAH_RANDOM_V1, HALF_CHEETAH_RANDOM_V2, HOPPER_EXPERT, HOPPER_EXPERT_V1, HOPPER_EXPERT_V2, HOPPER_MEDIUM, HOPPER_MEDIUM_EXPERT, HOPPER_MEDIUM_EXPERT_V1, HOPPER_MEDIUM_EXPERT_V2, HOPPER_MEDIUM_REPLAY, HOPPER_MEDIUM_REPLAY_V1, HOPPER_MEDIUM_REPLAY_V2, HOPPER_MEDIUM_V1, HOPPER_MEDIUM_V2, HOPPER_RANDOM, HOPPER_RANDOM_V1, HOPPER_RANDOM_V2, WALKER_EXPERT, WALKER_EXPERT_v1, WALKER_EXPERT_v2, WALKER_MEDIUM, WALKER_MEDIUM_EXPERT, WALKER_MEDIUM_EXPERT_V1, WALKER_MEDIUM_EXPERT_V2, WALKER_MEDIUM_REPLAY, WALKER_MEDIUM_REPLAY_V1, WALKER_MEDIUM_REPLAY_V2, WALKER_MEDIUM_v1, WALKER_MEDIUM_v2, WALKER_RANDOM, WALKER_RANDOM_v1, WALKER_RANDOM_v2
from benchmark.utils.load_dataset import load_dataset_from_env
import torch
import gym
import d4rl  # noqa


envs_with_preprocessing_functions = [
    HALF_CHEETAH_RANDOM,
    HALF_CHEETAH_MEDIUM,
    HALF_CHEETAH_EXPERT,
    HALF_CHEETAH_MEDIUM_REPLAY,
    HALF_CHEETAH_MEDIUM_EXPERT,
    HOPPER_RANDOM,
    HOPPER_MEDIUM,
    HOPPER_EXPERT,
    HOPPER_MEDIUM_REPLAY,
    HOPPER_MEDIUM_EXPERT,
    WALKER_RANDOM,
    WALKER_MEDIUM,
    WALKER_EXPERT,
    WALKER_MEDIUM_REPLAY,
    WALKER_MEDIUM_EXPERT,
    HALF_CHEETAH_RANDOM_V1,
    HALF_CHEETAH_MEDIUM_V1,
    HALF_CHEETAH_EXPERT_V1,
    HALF_CHEETAH_MEDIUM_REPLAY_V1,
    HALF_CHEETAH_MEDIUM_EXPERT_V1,
    HOPPER_RANDOM_V1,
    HOPPER_MEDIUM_V1,
    HOPPER_EXPERT_V1,
    HOPPER_MEDIUM_REPLAY_V1,
    HOPPER_MEDIUM_EXPERT_V1,
    WALKER_RANDOM_v1,
    WALKER_MEDIUM_v1,
    WALKER_EXPERT_v1,
    WALKER_MEDIUM_REPLAY_V1,
    WALKER_MEDIUM_EXPERT_V1,
    HALF_CHEETAH_RANDOM_V2,
    HALF_CHEETAH_MEDIUM_V2,
    HALF_CHEETAH_EXPERT_V2,
    HALF_CHEETAH_MEDIUM_REPLAY_V2,
    HALF_CHEETAH_MEDIUM_EXPERT_V2,
    HOPPER_RANDOM_V2,
    HOPPER_MEDIUM_V2,
    HOPPER_EXPERT_V2,
    HOPPER_MEDIUM_REPLAY_V2,
    HOPPER_MEDIUM_EXPERT_V2,
    WALKER_RANDOM_v2,
    WALKER_MEDIUM_v2,
    WALKER_EXPERT_v2,
    WALKER_MEDIUM_REPLAY_V2,
    WALKER_MEDIUM_EXPERT_V2,
]


def preprocess(mean, std, obs_act, detach=True):
    if detach:
        obs_act = obs_act.detach().clone()

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def get_preprocessing_function(env_name, device=''):
    if env_name not in envs_with_preprocessing_functions:
        return None

    env = gym.make(env_name)
    dataset = env.get_dataset()

    obs_act = torch.cat((torch.as_tensor(dataset['observations']), torch.as_tensor(dataset['actions'])), dim=1)

    mean = obs_act.mean(dim=0)
    std = obs_act.std(dim=0)

    if device != '':
        mean = mean.to(device)
        std = std.to(device)

    return partial(preprocess, mean, std)
