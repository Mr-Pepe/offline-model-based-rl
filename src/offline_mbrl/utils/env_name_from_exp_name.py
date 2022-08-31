from offline_mbrl.utils.envs import ALL_ENVS


def get_env_name_from_experiment_name(experiment_name: str) -> str:
    for env_name in ALL_ENVS:
        if env_name in experiment_name:
            return env_name

    raise ValueError(
        f"Failed to retrieve environment name from experiment name '{experiment_name}'. "
        "The experiment name did not contain any of the following environment "
        f"names: {ALL_ENVS}"
    )
