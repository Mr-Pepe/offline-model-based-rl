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


termination_functions = {
    'hopper': hopper_termination_fn,
}
