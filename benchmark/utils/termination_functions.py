from benchmark.utils.mazes import U_MAZE_WALLS
import torch


def hopper_termination_fn(next_obs):

    height = next_obs[:, :, 0]
    angle = next_obs[:, :, 1]
    not_done = torch.isfinite(next_obs).all(axis=-1) \
        * (next_obs[:, :, 1:] < 100).all(axis=-1) \
        * (height > .7) \
        * (torch.abs(angle) < .2)

    done = ~not_done
    done = done[:, :, None]
    return done


def half_cheetah_termination_fn(obs):

    done = torch.zeros((obs.shape[0], obs.shape[1], 1))
    return done


def walker2d_termination_fn(next_obs):

    height = next_obs[:, :, 0]
    angle = next_obs[:, :, 1]
    not_done = (height > 0.8) \
        * (height < 2.0) \
        * (angle > -1.0) \
        * (angle < 1.0)
    done = ~not_done
    done = done[:, :, None]
    return done


def umaze_termination_fn(next_obs):
    x = next_obs[:, :, 0]
    y = next_obs[:, :, 1]

    walls = U_MAZE_WALLS.to(x.device)

    done = torch.zeros((next_obs.shape[0], x[0].numel(), len(walls)))

    for i_wall, wall in enumerate(walls):
        done[:, :, i_wall] = (wall[0] < x) * (x < wall[1]) * \
            (wall[2] < y) * (y < wall[3])

    return done.sum(dim=2).reshape(next_obs.shape[0], -1, 1)


termination_functions = {
    'hopper': hopper_termination_fn,
    'half_cheetah': half_cheetah_termination_fn,
    'walker2d': walker2d_termination_fn,
    'umaze': umaze_termination_fn,
}

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
    'umaze': ['antmaze-umaze-v0',
              'antmaze-umaze-diverse-v0',
              'maze2d-umaze-dense-v1'],
}


def get_termination_function(env_name):
    for fn_name in function_to_names_mapping:
        if env_name in function_to_names_mapping[fn_name]:
            return termination_functions[fn_name]

    return None
