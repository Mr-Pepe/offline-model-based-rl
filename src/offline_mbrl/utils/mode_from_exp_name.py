from offline_mbrl.utils.modes import ALL_MODES


def get_mode_from_experiment_name(experiment_name: str) -> str:
    for mode in ALL_MODES:
        if mode in experiment_name:
            return mode

    raise ValueError(
        f"Failed to retrieve mode from experiment name '{experiment_name}'. "
        f"The experiment name did not contain any of the following modes: {ALL_MODES}"
    )
