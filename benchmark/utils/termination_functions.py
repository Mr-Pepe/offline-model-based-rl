import torch


def hopper_termination_fn(next_obs):

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = torch.isfinite(next_obs).all(axis=-1) \
        * (next_obs[:, 1:] < 100).all(axis=-1) \
        * (height > .7) \
        * (torch.abs(angle) < .2)

    done = ~not_done
    done = done[:, None]
    return done


def half_cheetah_termination_fn(obs):

    done = torch.as_tensor([False]).repeat(len(obs))
    done = done[:, None]
    return done


def walker2d_termination_fn(next_obs):

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.8) \
        * (height < 2.0) \
        * (angle > -1.0) \
        * (angle < 1.0)
    done = ~not_done
    done = done[:, None]
    return done


termination_functions = {
    'hopper': hopper_termination_fn,
    'half_cheetah': half_cheetah_termination_fn,
    'walker2d': walker2d_termination_fn,
}
