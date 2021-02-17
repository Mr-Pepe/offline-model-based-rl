from benchmark.utils.mazes import ANTMAZE_MEDIUM_DIVERSE_GOAL, ANTMAZE_UMAZE_DIVERSE_GOAL, ANTMAZE_UMAZE_GOAL
from benchmark.utils.modes import ALEATORIC_PARTITIONING, EPISTEMIC_PARTITIONING, EPISTEMIC_PENALTY, EXPLICIT_PARTITIONING, EXPLICIT_PENALTY, MODES, ALEATORIC_PENALTY, PARTITIONING_MODES, PENALTY_MODES
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
               HOPPER_MEDIUM_EXPERT_V1]

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
                     HALF_CHEETAH_MEDIUM_EXPERT_V1]

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
               WALKER_MEDIUM_EXPERT_V1]

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

REWARD_SPANS = {
    HOPPER_MEDIUM: 5.2582,
    HOPPER_MEDIUM_REPLAY: 6.2268,
    HOPPER_MEDIUM_EXPERT: 7.5698,
    WALKER_MEDIUM: 14.5130,
    WALKER_MEDIUM_REPLAY: 9.5062,
    WALKER_MEDIUM_EXPERT: 16.0128,
}

# (max, mean, std)
ALEATORIC_UNCERTAINTIES = {
    HOPPER_MEDIUM: (2.497604, 0.165585, 0.403048),
    HOPPER_MEDIUM_REPLAY: (2.546367, 0.492017, 0.546482),
    HOPPER_MEDIUM_EXPERT: (1.414543, 0.317964, 0.368814),
    WALKER_MEDIUM: (22.540873, 11.234148, 7.134698),
    WALKER_MEDIUM_REPLAY: (25.140537, 11.956196, 8.148096),
    WALKER_MEDIUM_EXPERT: (21.668726, 13.161317, 6.534163),
}

EPISTEMIC_UNCERTAINTIES = {
    HOPPER_MEDIUM: (7.977843, 0.303133, 0.482348),
    HOPPER_MEDIUM_REPLAY: (8.079539, 0.554741, 0.520457),
    HOPPER_MEDIUM_EXPERT: (9.838729, 0.543635, 0.588447),
    WALKER_MEDIUM: (27.487648, 4.275870, 2.892133),
    WALKER_MEDIUM_REPLAY: (27.632509, 3.033420, 2.194959),
    WALKER_MEDIUM_EXPERT: (28.805645, 4.684679, 2.561787),
}

EXPLICIT_UNCERTAINTIES = {
    HOPPER_MEDIUM: (0.388037, 0.000025, 0.001877),
    HOPPER_MEDIUM_REPLAY: (0.857102, 0.000059, 0.003437),
    HOPPER_MEDIUM_EXPERT: (0.999997, 0.000052, 0.003960),
    WALKER_MEDIUM: (0.844097, 0.000163, 0.005674),
    WALKER_MEDIUM_REPLAY: (0.415466, 0.000083, 0.003571),
    WALKER_MEDIUM_EXPERT: (1.000000, 0.000184, 0.007116),
}

# Hyperparameters
# (rollouts, rollout length, pessimism/ OOD threshold)
HYPERPARAMS = {
    ALEATORIC_PARTITIONING: {
        HOPPER_MEDIUM: (57, 42, 0.61826)
    },
    EPISTEMIC_PARTITIONING: {
        HOPPER_MEDIUM: (91, 40, 2.7911)
    },
    EXPLICIT_PARTITIONING: {
        HOPPER_MEDIUM: (99, 38, 0.044925)
    },
    ALEATORIC_PENALTY: {
        HOPPER_MEDIUM: (52, 47, 1.7582)
    },
    EPISTEMIC_PENALTY: {
        HOPPER_MEDIUM: (39, 41, 0.29138)
    },
    EXPLICIT_PENALTY: {
        HOPPER_MEDIUM: (10, 31, 13.551)
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
