from benchmark.utils.mazes import ANTMAZE_MEDIUM_DIVERSE_GOAL, ANTMAZE_UMAZE_DIVERSE_GOAL, ANTMAZE_UMAZE_GOAL
from benchmark.utils.modes import ALEATORIC_PARTITIONING, EPISTEMIC_PARTITIONING, EPISTEMIC_PENALTY, EXPLICIT_PARTITIONING, EXPLICIT_PENALTY, ALEATORIC_PENALTY
import gym
import d4rl  # noqa

ANTMAZE_UMAZE_ENVS = ['antmaze-umaze-v0',
                      'antmaze-umaze-diverse-v0']

MAZE2D_UMAZE_ENVS = ['maze2d-umaze-v1']

HOPPER_ORIGINAL = 'Hopper-v2'
HOPPER_RANDOM = 'hopper-random-v0'
HOPPER_MEDIUM = 'hopper-medium-v0'
HOPPER_EXPERT = 'hopper-expert-v0'
HOPPER_MEDIUM_REPLAY = 'hopper-medium-replay-v0'
HOPPER_MEDIUM_EXPERT = 'hopper-medium-expert-v0'

HOPPER_RANDOM_V1 = 'hopper-random-v1'
HOPPER_MEDIUM_V1 = 'hopper-medium-v1'
HOPPER_EXPERT_V1 = 'hopper-expert-v1'
HOPPER_MEDIUM_REPLAY_V1 = 'hopper-medium-replay-v1'
HOPPER_MEDIUM_EXPERT_V1 = 'hopper-medium-expert-v1'

HOPPER_RANDOM_V2 = 'hopper-random-v2'
HOPPER_MEDIUM_V2 = 'hopper-medium-v2'
HOPPER_EXPERT_V2 = 'hopper-expert-v2'
HOPPER_MEDIUM_REPLAY_V2 = 'hopper-medium-replay-v2'
HOPPER_MEDIUM_EXPERT_V2 = 'hopper-medium-expert-v2'

HOPPER_ENVS = [HOPPER_ORIGINAL,
               HOPPER_RANDOM,
               HOPPER_MEDIUM,
               HOPPER_EXPERT,
               HOPPER_MEDIUM_REPLAY,
               HOPPER_MEDIUM_EXPERT,
               HOPPER_RANDOM_V1,
               HOPPER_MEDIUM_V1,
               HOPPER_EXPERT_V1,
               HOPPER_MEDIUM_REPLAY_V1,
               HOPPER_MEDIUM_EXPERT_V1,
               HOPPER_RANDOM_V2,
               HOPPER_MEDIUM_V2,
               HOPPER_EXPERT_V2,
               HOPPER_MEDIUM_REPLAY_V2,
               HOPPER_MEDIUM_EXPERT_V2]

HALF_CHEETAH_ORIGINAL = 'HalfCheetah-v2'
HALF_CHEETAH_RANDOM = 'halfcheetah-random-v0'
HALF_CHEETAH_MEDIUM = 'halfcheetah-medium-v0'
HALF_CHEETAH_EXPERT = 'halfcheetah-expert-v0'
HALF_CHEETAH_MEDIUM_REPLAY = 'halfcheetah-medium-replay-v0'
HALF_CHEETAH_MEDIUM_EXPERT = 'halfcheetah-medium-expert-v0'

HALF_CHEETAH_RANDOM_V1 = 'halfcheetah-random-v1'
HALF_CHEETAH_MEDIUM_V1 = 'halfcheetah-medium-v1'
HALF_CHEETAH_EXPERT_V1 = 'halfcheetah-expert-v1'
HALF_CHEETAH_MEDIUM_REPLAY_V1 = 'halfcheetah-medium-replay-v1'
HALF_CHEETAH_MEDIUM_EXPERT_V1 = 'halfcheetah-medium-expert-v1'

HALF_CHEETAH_RANDOM_V2 = 'halfcheetah-random-v2'
HALF_CHEETAH_MEDIUM_V2 = 'halfcheetah-medium-v2'
HALF_CHEETAH_EXPERT_V2 = 'halfcheetah-expert-v2'
HALF_CHEETAH_MEDIUM_REPLAY_V2 = 'halfcheetah-medium-replay-v2'
HALF_CHEETAH_MEDIUM_EXPERT_V2 = 'halfcheetah-medium-expert-v2'

HALF_CHEETAH_ENVS = [HALF_CHEETAH_ORIGINAL,
                     HALF_CHEETAH_RANDOM,
                     HALF_CHEETAH_MEDIUM,
                     HALF_CHEETAH_EXPERT,
                     HALF_CHEETAH_MEDIUM_REPLAY,
                     HALF_CHEETAH_MEDIUM_EXPERT,
                     HALF_CHEETAH_RANDOM_V1,
                     HALF_CHEETAH_MEDIUM_V1,
                     HALF_CHEETAH_EXPERT_V1,
                     HALF_CHEETAH_MEDIUM_REPLAY_V1,
                     HALF_CHEETAH_MEDIUM_EXPERT_V1,
                     HALF_CHEETAH_RANDOM_V2,
                     HALF_CHEETAH_MEDIUM_V2,
                     HALF_CHEETAH_EXPERT_V2,
                     HALF_CHEETAH_MEDIUM_REPLAY_V2,
                     HALF_CHEETAH_MEDIUM_EXPERT_V2]

WALKER_ORIGINAL = 'Walker2d-v2'
WALKER_RANDOM = 'walker2d-random-v0'
WALKER_MEDIUM = 'walker2d-medium-v0'
WALKER_EXPERT = 'walker2d-expert-v0'
WALKER_MEDIUM_REPLAY = 'walker2d-medium-replay-v0'
WALKER_MEDIUM_EXPERT = 'walker2d-medium-expert-v0'

WALKER_RANDOM_v1 = 'walker2d-random-v1'
WALKER_MEDIUM_v1 = 'walker2d-medium-v1'
WALKER_EXPERT_v1 = 'walker2d-expert-v1'
WALKER_MEDIUM_REPLAY_V1 = 'walker2d-medium-replay-v1'
WALKER_MEDIUM_EXPERT_V1 = 'walker2d-medium-expert-v1'

WALKER_RANDOM_v2 = 'walker2d-random-v2'
WALKER_MEDIUM_v2 = 'walker2d-medium-v2'
WALKER_EXPERT_v2 = 'walker2d-expert-v2'
WALKER_MEDIUM_REPLAY_V2 = 'walker2d-medium-replay-v2'
WALKER_MEDIUM_EXPERT_V2 = 'walker2d-medium-expert-v2'


WALKER_ENVS = [WALKER_ORIGINAL,
               WALKER_RANDOM,
               WALKER_MEDIUM,
               WALKER_EXPERT,
               WALKER_MEDIUM_REPLAY,
               WALKER_MEDIUM_EXPERT,
               WALKER_RANDOM_v1,
               WALKER_MEDIUM_v1,
               WALKER_EXPERT_v1,
               WALKER_MEDIUM_REPLAY_V1,
               WALKER_MEDIUM_EXPERT_V1,
               WALKER_RANDOM_v2,
               WALKER_MEDIUM_v2,
               WALKER_EXPERT_v2,
               WALKER_MEDIUM_REPLAY_V2,
               WALKER_MEDIUM_EXPERT_V2]

ANTMAZE_MEDIUM_ENVS = ['antmaze-medium-diverse-v0',
                       'antmaze-medium-play-v0']

ENV_CATEGORIES = {
    'hopper': HOPPER_ENVS,
    'half_cheetah': HALF_CHEETAH_ENVS,
    'walker2d': WALKER_ENVS,
    'antmaze_umaze': ANTMAZE_UMAZE_ENVS,
    'maze2d_umaze': MAZE2D_UMAZE_ENVS,
    'antmaze_medium': ANTMAZE_MEDIUM_ENVS,
}

# Hyperparameters
# (rollouts, rollout length, pessimism/ OOD threshold)
HYPERPARAMS = {
    ALEATORIC_PARTITIONING: {
        HOPPER_MEDIUM_V2: (50, 19, 0.0131),
        HOPPER_MEDIUM_REPLAY_V2: (50, 9, 0.0483),
        HOPPER_MEDIUM_EXPERT_V2: (50, 9, 0.0473),
        WALKER_MEDIUM_REPLAY_V2: (50, 20, 1.1510)
    },
    EPISTEMIC_PARTITIONING: {
        HOPPER_MEDIUM_V2: (50, 13, 1.0257),
        HOPPER_MEDIUM_REPLAY_V2: (50, 20, 0.599),
        HOPPER_MEDIUM_EXPERT_V2: (50, 20, 1.0881),
        WALKER_MEDIUM_v2: (50, 11, 1.5975),
        WALKER_MEDIUM_REPLAY_V2: (50, 4, 3.12812),
    },
    EXPLICIT_PARTITIONING: {
    },
    ALEATORIC_PENALTY: {
        HOPPER_MEDIUM_V2: (50, 13, 48.34),
        HOPPER_MEDIUM_REPLAY_V2: (50, 8, 48.06),
        HOPPER_MEDIUM_EXPERT_V2: (50, 10, 140.01),
        WALKER_MEDIUM_REPLAY_V2: (50, 20, 1.32213)
    },
    EPISTEMIC_PENALTY: {
        HOPPER_MEDIUM_V2: (50, 19, 1e-10),
        HOPPER_MEDIUM_REPLAY_V2: (50, 9, 2.2202),
        HOPPER_MEDIUM_EXPERT_V2: (50, 15, 6.0918),
        WALKER_MEDIUM_v2: (50, 11, 1.19685),
        WALKER_MEDIUM_REPLAY_V2: (50, 6, 0.53859),
    },
    EXPLICIT_PENALTY: {
    }
}


def antmaze_umaze_test_env():
    env = gym.make('antmaze-umaze-v0')
    env.set_target(ANTMAZE_UMAZE_GOAL)
    return env


def antmaze_umaze_diverse_test_env():
    env = gym.make('antmaze-umaze-v0')
    env.set_target(ANTMAZE_UMAZE_DIVERSE_GOAL)
    return env


def antmaze_medium_diverse_test_env():
    env = gym.make('antmaze-medium-diverse-v0')
    env.set_target(ANTMAZE_MEDIUM_DIVERSE_GOAL)
    return env


TEST_ENV_MAPPING = {
    'antmaze-umaze-v0': antmaze_umaze_test_env,
    'antmaze-umaze-diverse-v0': antmaze_umaze_diverse_test_env,
    'antmaze-medium-diverse-v0': antmaze_medium_diverse_test_env,
    'antmaze-medium-play-v0': None
}


def get_test_env(env_name):
    if env_name in TEST_ENV_MAPPING:
        return TEST_ENV_MAPPING[env_name]()
    else:
        return gym.make(env_name)
