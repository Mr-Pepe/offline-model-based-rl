import argparse
from pathlib import Path
from typing import Optional

from offline_mbrl.schemas import (
    BehavioralCloningConfiguration,
    EnvironmentModelConfiguration,
    EpochLoggerConfiguration,
    SACConfiguration,
    TrainerConfiguration,
)
from offline_mbrl.train import Trainer
from offline_mbrl.user_config import DATA_DIR, MODELS_DIR
from offline_mbrl.utils.get_experiment_name import get_experiment_name
from offline_mbrl.utils.hyperparameters import HYPERPARAMS
from offline_mbrl.utils.modes import (
    ALL_MODES,
    BEHAVIORAL_CLONING,
    MODEL_BASED_MODES,
    OFFLINE_MODES,
    PARTITIONING_MODES,
    PENALTY_MODES,
)
from offline_mbrl.utils.preprocessing import get_preprocessing_function
from offline_mbrl.utils.setup_logger_kwargs import setup_logger_kwargs
from offline_mbrl.utils.termination_functions import get_termination_function


def main(args: argparse.Namespace):

    trainer_config = TrainerConfiguration()

    trainer_config.env_name = args.env_name

    if args.mode not in ALL_MODES:
        raise ValueError(f"Unknown mode: {args.mode}")

    preprocessing_function = get_preprocessing_function(trainer_config.env_name)
    termination_function = get_termination_function(trainer_config.env_name)

    env_model_config: Optional[
        EnvironmentModelConfiguration
    ] = EnvironmentModelConfiguration(
        type="probabilistic",
        n_networks=8,
        preprocessing_function=preprocessing_function,
        termination_function=termination_function,
    )

    if args.mode == BEHAVIORAL_CLONING:
        agent_config = BehavioralCloningConfiguration(
            preprocessing_function=preprocessing_function
        )
    else:
        agent_config = SACConfiguration(preprocessing_function=preprocessing_function)

    if args.mode in MODEL_BASED_MODES:
        trainer_config.use_env_model = True

        pretrained_model_path: Path = Path(MODELS_DIR) / (args.env_name + "-model.pt")

        if args.mode in OFFLINE_MODES and pretrained_model_path.is_file():
            trainer_config.pretrained_env_model_path = pretrained_model_path

    if args.mode in OFFLINE_MODES:
        trainer_config.online_epochs = 0
        trainer_config.offline_epochs = 100
        trainer_config.n_samples_from_dataset = None

    if args.mode in PARTITIONING_MODES:
        (
            trainer_config.max_virtual_rollout_length,
            env_model_config.ood_threshold,
        ) = HYPERPARAMS[args.mode][args.env_name]
        env_model_config.pessimism = 0
    elif args.mode in PENALTY_MODES:
        (
            trainer_config.max_virtual_rollout_length,
            env_model_config.pessimism,
        ) = HYPERPARAMS[args.mode][args.env_name]
        env_model_config.ood_threshold = 0

    logger_config = EpochLoggerConfiguration(
        **setup_logger_kwargs(
            exp_name=get_experiment_name(trainer_config, args.mode),
            seed=trainer_config.seed,
            data_dir=DATA_DIR,
        )
    )

    trainer = Trainer(trainer_config, agent_config, env_model_config, logger_config)
    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    main(parser.parse_args())
