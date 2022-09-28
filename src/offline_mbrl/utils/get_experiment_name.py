from offline_mbrl.schemas import TrainerConfiguration


def get_experiment_name(config: TrainerConfiguration, mode: str) -> str:
    exp_name = f"{config.env_name}-{mode}"

    return exp_name
