import torch

from offline_mbrl.utils.envs import ENV_CATEGORIES


def postprocess_hopper(next_obs=None, **_):
    next_obs = next_obs.detach().clone()

    height = next_obs[:, :, 0]
    angle = next_obs[:, :, 1]
    not_done = (
        torch.isfinite(next_obs).all(axis=-1)
        * (next_obs[:, :, 1:] < 100).all(axis=-1)
        * (height > 0.7)
        * (torch.abs(angle) < 0.2)
    )

    done = ~not_done
    done = done[:, :, None]
    return {"dones": done}


def postprocess_half_cheetah(next_obs=None, **_):
    next_obs = next_obs.detach().clone()
    done = torch.zeros((next_obs.shape[0], next_obs.shape[1], 1))
    return {"dones": done}


def postprocess_walker2d(next_obs=None, **_):
    next_obs = next_obs.detach().clone()
    height = next_obs[:, :, 0]
    angle = next_obs[:, :, 1]
    not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
    done = ~not_done
    done = done[:, :, None]
    return {"dones": done}


postprocessing_functions = {
    "hopper": postprocess_hopper,
    "half_cheetah": postprocess_half_cheetah,
    "walker2d": postprocess_walker2d,
}


def get_postprocessing_function(env_name):
    for fn_name in ENV_CATEGORIES:
        if env_name in ENV_CATEGORIES[fn_name]:
            return postprocessing_functions[fn_name]

    return None
