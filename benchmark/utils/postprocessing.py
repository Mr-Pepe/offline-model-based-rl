from benchmark.utils.envs import ANTMAZE_UMAZE_ENVS, ENV_CATEGORIES, HALF_CHEETAH_ENVS, HOPPER_ENVS, MAZE2D_UMAZE_ENVS, WALKER_ENVS
from benchmark.utils.mazes import ANTMAZE_ANT_RADIUS, ANTMAZE_MEDIUM_MAX, ANTMAZE_MEDIUM_MIN, ANTMAZE_MEDIUM_WALLS_WITHOUT_OUTSIDE, ANTMAZE_MEDIUM_WALLS_WITHOUT_OUTSIDE_CUDA, ANTMAZE_UMAZE_MAX, ANTMAZE_UMAZE_MIN, ANTMAZE_UMAZE_WALLS_WITHOUT_OUTSIDE, ANTMAZE_UMAZE_WALLS_WITHOUT_OUTSIDE_CUDA, MAZE2D_POINT_RADIUS, \
    MAZE2D_UMAZE_WALLS
import torch


def postprocess_hopper(next_obs=None, **_):
    next_obs = next_obs.detach().clone()

    height = next_obs[:, :, 0]
    angle = next_obs[:, :, 1]
    not_done = torch.isfinite(next_obs).all(axis=-1) \
        * (next_obs[:, :, 1:] < 100).all(axis=-1) \
        * (height > .7) \
        * (torch.abs(angle) < .2)

    done = ~not_done
    done = done[:, :, None]
    return {'dones': done}


def postprocess_half_cheetah(next_obs=None, **_):
    next_obs = next_obs.detach().clone()
    done = torch.zeros((next_obs.shape[0], next_obs.shape[1], 1))
    return {'dones': done}


def postprocess_walker2d(next_obs=None, **_):
    next_obs = next_obs.detach().clone()
    height = next_obs[:, :, 0]
    angle = next_obs[:, :, 1]
    not_done = (height > 0.8) \
        * (height < 2.0) \
        * (angle > -1.0) \
        * (angle < 1.0)
    done = ~not_done
    done = done[:, :, None]
    return {'dones': done}


def postprocess_antmaze_umaze(next_obs=None, means=None, logvars=None, **_):
    x = next_obs[:, :, 0]
    y = next_obs[:, :, 1]

    if x.device.type == 'cpu':
        walls = ANTMAZE_UMAZE_WALLS_WITHOUT_OUTSIDE
    else:
        walls = ANTMAZE_UMAZE_WALLS_WITHOUT_OUTSIDE_CUDA

    maze_min = ANTMAZE_UMAZE_MIN + ANTMAZE_ANT_RADIUS
    maze_max = ANTMAZE_UMAZE_MAX - ANTMAZE_ANT_RADIUS

    collision = \
        (((walls[:, 0] <= x.unsqueeze(-1) + ANTMAZE_ANT_RADIUS) *
          (walls[:, 1] > x.unsqueeze(-1) - ANTMAZE_ANT_RADIUS) *
          (walls[:, 2] <= y.unsqueeze(-1) + ANTMAZE_ANT_RADIUS) *
          (walls[:, 3] > y.unsqueeze(-1) - ANTMAZE_ANT_RADIUS)).sum(dim=2) +
         (maze_max <= x) +
         (x <= maze_min) +
         (maze_max <= y) +
         (y <= maze_min)).to(x.device)

    collision = collision > 0

    notdone = ((torch.isfinite(next_obs).all(dim=2) *
                (next_obs[:, :, 2] >= 0.2) *
                (next_obs[:, :, 2] <= 1.0)))
    done = ~notdone
    done = torch.logical_or(done, collision)

    return {'dones': done.unsqueeze(-1)}


def postprocess_antmaze_medium(next_obs=None, means=None, logvars=None, **_):
    x = next_obs[:, :, 0]
    y = next_obs[:, :, 1]

    if x.device.type == 'cpu':
        walls = ANTMAZE_MEDIUM_WALLS_WITHOUT_OUTSIDE
    else:
        walls = ANTMAZE_MEDIUM_WALLS_WITHOUT_OUTSIDE_CUDA

    maze_min = ANTMAZE_MEDIUM_MIN + ANTMAZE_ANT_RADIUS
    maze_max = ANTMAZE_MEDIUM_MAX - ANTMAZE_ANT_RADIUS

    collision = \
        (((walls[:, 0] <= x.unsqueeze(-1) + ANTMAZE_ANT_RADIUS) *
          (walls[:, 1] > x.unsqueeze(-1) - ANTMAZE_ANT_RADIUS) *
          (walls[:, 2] <= y.unsqueeze(-1) + ANTMAZE_ANT_RADIUS) *
          (walls[:, 3] > y.unsqueeze(-1) - ANTMAZE_ANT_RADIUS)).sum(dim=2) +
         (maze_max <= x) +
         (x <= maze_min) +
         (maze_max <= y) +
         (y <= maze_min)).to(x.device)

    collision = collision > 0

    notdone = ((torch.isfinite(next_obs).all(dim=2) *
                (next_obs[:, :, 2] >= 0.2) *
                (next_obs[:, :, 2] <= 1.0)))
    done = ~notdone
    done = torch.logical_or(done, collision)

    return {'dones': done.unsqueeze(-1)}


def postprocess_maze2d_umaze(next_obs=None, means=None, logvars=None, **_):
    next_obs = next_obs.detach().clone()
    if means is not None:
        means = means.detach().clone()
    if logvars is not None:
        logvars = logvars.detach().clone()

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

    collision = collision.sum(dim=2) > 0

    if collision.any():
        for i_network in range(next_obs.shape[0]):
            # Reset all means and logvars to prevent exploration here
            if means is not None:
                means[i_network][collision[i_network]] = 0

            if logvars is not None:
                logvars[i_network][collision[i_network]] = -20

    return {'dones': collision.unsqueeze(-1).to(x.device),
            'means': means,
            'logvars': logvars}


postprocessing_functions = {
    'hopper': postprocess_hopper,
    'half_cheetah': postprocess_half_cheetah,
    'walker2d': postprocess_walker2d,
    'antmaze_umaze': postprocess_antmaze_umaze,
    'maze2d_umaze': postprocess_maze2d_umaze,
    'antmaze_medium': postprocess_antmaze_medium
}


def get_postprocessing_function(env_name):
    for fn_name in ENV_CATEGORIES:
        if env_name in ENV_CATEGORIES[fn_name]:
            return postprocessing_functions[fn_name]

    return None
