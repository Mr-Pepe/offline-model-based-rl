from benchmark.utils.mazes import ANTMAZE_UMAZE_WALLS, MAZE2D_POINT_RADIUS, \
    MAZE2D_UMAZE_WALLS
import torch


def hopper_termination_fn(next_obs=None, **_):

    height = next_obs[:, :, 0]
    angle = next_obs[:, :, 1]
    not_done = torch.isfinite(next_obs).all(axis=-1) \
        * (next_obs[:, :, 1:] < 100).all(axis=-1) \
        * (height > .7) \
        * (torch.abs(angle) < .2)

    done = ~not_done
    done = done[:, :, None]
    return done


def half_cheetah_termination_fn(next_obs=None, **_):

    done = torch.zeros((next_obs.shape[0], next_obs.shape[1], 1))
    return done


def walker2d_termination_fn(next_obs=None, **_):

    height = next_obs[:, :, 0]
    angle = next_obs[:, :, 1]
    not_done = (height > 0.8) \
        * (height < 2.0) \
        * (angle > -1.0) \
        * (angle < 1.0)
    done = ~not_done
    done = done[:, :, None]
    return done


def antmaze_umaze_termination_fn(next_obs=None, **_):
    x = next_obs[:, :, 0]
    y = next_obs[:, :, 1]

    walls = ANTMAZE_UMAZE_WALLS.to(x.device)

    done = torch.zeros((next_obs.shape[0], x[0].numel(), len(walls)+1))

    for i_wall, wall in enumerate(walls):
        done[:, :, i_wall] = (wall[0] <= x) * (x <= wall[1]) * \
            (wall[2] <= y) * (y <= wall[3])

    x_min = walls[:, 0].min()
    x_max = walls[:, 1].max()
    y_min = walls[:, 2].min()
    y_max = walls[:, 3].max()

    done[:, :, -1] = (x_max <= x) + (x <= x_min) + (y_max <= y) + (y <= y_min)

    return done.sum(dim=2).reshape(next_obs.shape[0], -1, 1)


def maze2d_umaze_termination_fn(next_obs=None, obs=None, **_):
    x = next_obs[:, :, 0]
    y = next_obs[:, :, 1]

    walls = MAZE2D_UMAZE_WALLS.to(x.device)

    collision = torch.zeros((next_obs.shape[0], x[0].numel(), len(walls)+1))

    for i_wall, wall in enumerate(walls):
        collision[:, :, i_wall] = \
            (wall[0] <= x + MAZE2D_POINT_RADIUS) * \
            (wall[1] > x - MAZE2D_POINT_RADIUS) * \
            (wall[2] <= y + MAZE2D_POINT_RADIUS) * \
            (wall[3] > y - MAZE2D_POINT_RADIUS)

    x_min = walls[:, 0].min()
    x_max = walls[:, 1].max()
    y_min = walls[:, 2].min()
    y_max = walls[:, 3].max()

    collision[:, :, -1] = (x_max <= x) + (x <= x_min) + \
        (y_max <= y) + (y <= y_min)

    collision = collision.sum(dim=2)

    for i_network in range(next_obs.shape[0]):
        next_obs[i_network][collision[i_network] > 0] = \
            obs[collision[i_network] > 0].detach().clone()

    return torch.zeros((next_obs.shape[0], next_obs.shape[1], 1)).to(x.device)


termination_functions = {
    'hopper': hopper_termination_fn,
    'half_cheetah': half_cheetah_termination_fn,
    'walker2d': walker2d_termination_fn,
    'antmaze_umaze': antmaze_umaze_termination_fn,
    'maze2d_umaze': maze2d_umaze_termination_fn,
}

ANTMAZE_UMAZE_ENVS = ['antmaze-umaze-v0',
                      'antmaze-umaze-diverse-v0']

MAZE2D_UMAZE_ENVS = ['maze2d-umaze-v1']

function_to_names_mapping = {
    'hopper': ['Hopper-v2',
               'hopper-random-v0',
               'hopper-medium-v0',
               'hopper-expert-v0',
               'hopper-medium-replay-v0',
               'hopper-medium-expert-v0'],
    'half_cheetah': ['HalfCheetah-v2',
                     'halfcheetah-random-v0',
                     'halfcheetah-medium-v0',
                     'halfcheetah-expert-v0',
                     'halfcheetah-medium-replay-v0',
                     'halfcheetah-medium-expert-v0'],
    'walker2d': ['Walker2d-v2',
                 'walker2d-random-v0',
                 'walker2d-medium-v0',
                 'walker2d-expert-v0',
                 'walker2d-medium-replay-v0',
                 'walker2d-medium-expert-v0',
                 ],
    'antmaze_umaze': ANTMAZE_UMAZE_ENVS,
    'maze2d_umaze': MAZE2D_UMAZE_ENVS,
}


def get_termination_function(env_name):
    for fn_name in function_to_names_mapping:
        if env_name in function_to_names_mapping[fn_name]:
            return termination_functions[fn_name]

    return None
