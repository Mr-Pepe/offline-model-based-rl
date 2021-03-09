from benchmark.utils.envs import HALF_CHEETAH_ENVS, HOPPER_ENVS, WALKER_ENVS


def get_env_name(exp_name):
    for env_name in HOPPER_ENVS + HALF_CHEETAH_ENVS + WALKER_ENVS:
        if env_name in exp_name:
            return env_name

    return None
