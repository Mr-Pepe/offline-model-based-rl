ANTMAZE_UMAZE_ENVS = ['antmaze-umaze-v0',
                      'antmaze-umaze-diverse-v0']

MAZE2D_UMAZE_ENVS = ['maze2d-umaze-v1']


HOPPER_ENVS = ['Hopper-v2',
               'hopper-random-v0',
               'hopper-medium-v0',
               'hopper-expert-v0',
               'hopper-medium-replay-v0',
               'hopper-medium-expert-v0']

HALF_CHEETAH_ENVS = ['HalfCheetah-v2',
                     'halfcheetah-random-v0',
                     'halfcheetah-medium-v0',
                     'halfcheetah-expert-v0',
                     'halfcheetah-medium-replay-v0',
                     'halfcheetah-medium-expert-v0']

WALKER_ENVS = ['Walker2d-v2',
               'walker2d-random-v0',
               'walker2d-medium-v0',
               'walker2d-expert-v0',
               'walker2d-medium-replay-v0',
               'walker2d-medium-expert-v0']

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
