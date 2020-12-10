from benchmark.utils.envs import ANTMAZE_UMAZE_ENVS, ENV_CATEGORIES, HALF_CHEETAH_ENVS, HOPPER_ENVS, MAZE2D_UMAZE_ENVS, WALKER_ENVS
from benchmark.utils.mazes import ANTMAZE_ANT_RADIUS, ANTMAZE_UMAZE_WALLS, MAZE2D_POINT_RADIUS, \
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


def antmaze_umaze_termination_fn(next_obs=None, obs=None, means=None, logvars=None, **_):
    x = next_obs[:, :, 0]
    y = next_obs[:, :, 1]

    walls = ANTMAZE_UMAZE_WALLS.to(x.device)

    collision = torch.zeros((next_obs.shape[0], x[0].numel(), len(walls)+1))

    for i_wall, wall in enumerate(walls):
        collision[:, :, i_wall] = \
            (wall[0] <= x + ANTMAZE_ANT_RADIUS) * \
            (wall[1] > x - ANTMAZE_ANT_RADIUS) * \
            (wall[2] <= y + ANTMAZE_ANT_RADIUS) * \
            (wall[3] > y - ANTMAZE_ANT_RADIUS)

    x_min = walls[:, 0].min()
    x_max = walls[:, 1].max()
    y_min = walls[:, 2].min()
    y_max = walls[:, 3].max()

    collision[:, :, -1] = (x_max <= x) + (x <= x_min) + \
        (y_max <= y) + (y <= y_min)

    collision = collision.sum(dim=2) > 0

    if collision.any():
        for i_network in range(next_obs.shape[0]):
            next_obs[i_network][collision[i_network]] = \
                obs[collision[i_network]].detach().clone()

            # Reset all means and logvars to prevent exploration here
            if means is not None:
                means[i_network][collision[i_network]] = \
                    next_obs[i_network][collision[i_network]]

            if logvars is not None:
                logvars[i_network][collision[i_network]] = -100

    notdone = torch.stack((torch.isfinite(next_obs).all(dim=2),
                           next_obs[:, :, 2] >= 0.2,
                           next_obs[:, :, 2] <= 1.0)).all(dim=0)
    done = ~notdone
    done = torch.logical_or(done, collision.to(x.device))

    return done.unsqueeze(-1)


def maze2d_umaze_termination_fn(next_obs=None, obs=None, means=None, logvars=None, **_):
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
            next_obs[i_network][collision[i_network]] = \
                obs[collision[i_network]].detach().clone()

            # Reset all means and logvars to prevent exploration here
            if means is not None:
                means[i_network][collision[i_network]] = \
                    next_obs[i_network][collision[i_network]]

            if logvars is not None:
                logvars[i_network][collision[i_network]] = -100

    return collision.unsqueeze(-1).to(x.device)


termination_functions = {
    'hopper': hopper_termination_fn,
    'half_cheetah': half_cheetah_termination_fn,
    'walker2d': walker2d_termination_fn,
    'antmaze_umaze': antmaze_umaze_termination_fn,
    'maze2d_umaze': maze2d_umaze_termination_fn,
}


def get_termination_function(env_name):
    for fn_name in ENV_CATEGORIES:
        if env_name in ENV_CATEGORIES[fn_name]:
            return termination_functions[fn_name]

    return None
